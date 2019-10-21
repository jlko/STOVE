"""Contains EvaluationHelper class.

Is used to analyse finished runs.
For example of usage see evaluation_graphics.ipynb.
"""

import os
from itertools import product

import pandas as pd
from matplotlib import pyplot as plt
from SortedSet.sorted_set import SortedSet


class EvaluationHelper:
    """Analyse performance of runs in experiment_dir."""

    def __init__(self, runs, user=None, zs=None):
        """Read given runs."""
        self.runs = SortedSet(runs)
        self.perfs, self.confs = self.read_runs(runs)
        if zs is None:
            self.zs = ['z', 'rollout', 'z_sup', 'z_dyn']
        else:
            self.zs = zs

        self.add_size()
        self.color_cylce = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # read rollout errors as property, see below
        self._rollouts = None

        frames = ['pred', 'recon', 'total']
        self.supairvised_groups = list(product(['vin_true'], frames, [True, False]))\
            + list(product(['vin_sup', 'sup_true'], frames, [False]))

        # for printing of gif_wall location
        if user is None:
            self.user = 'user'
        else:
            self.user = user

    def read_runs(self, runs):
        """Read performance metrics and configs for runs."""
        perfs = {}
        confs = {}
        for run in runs:
            try:
                perfs[run] = pd.read_csv(os.path.join(
                    run, 'performance.csv')).set_index('step')
                perfs[run]['type'] = perfs[run]['type'].str.strip()
                confs[run] = pd.read_csv(os.path.join(
                    run, 'config.txt')).set_index('setting name')

            except Exception as e:
                print(e)
                print("Could not read run {}.".format(run))
                print("Deleting from self.runs")
                self.runs = self.runs - {run}

        return perfs, confs

    def update_runs(self, extras):
        """Read extra runs from different experiment_dir.

        Use to compare against precious experiments.
        """
        extra_perfs, extra_confs = self.read_runs(extras)
        self.perfs.update(extra_perfs)
        self.confs.update(extra_confs)
        self.runs = self.runs.union(extras)
        self.add_size(extras)

    def add_size(self, runs=None):
        """Add size and duration of performance.csv to config.

        This adds 'progress' of run.
        """
        if runs is None:
            runs = list(self.runs)

        for run in runs:
            perf = self.perfs[run]
            run_size = perf.index.max()
            self.confs[run] = self.confs[run].append(
                pd.Series(data={' setting value': run_size}, name='max_step'))
            # time in hours
            duration = perf.time.max()/60/60
            self.confs[run] = self.confs[run].append(
                pd.Series(data={' setting value': duration}, name='duration'))

    def show_differences(self, runs=None):
        """Show differences of configs between runs.

        Immediately see what config values are different from default for this
        experiment dir.
        """
        if runs is None:
            runs = self.runs

        configs = pd.DataFrame()

        # Stack all Configs and show differences
        for run in runs:
            conf = self.confs[run]
            tmp = conf.T
            tmp['run'] = run
            tmp = tmp.set_index('run')
            configs = configs.append(tmp, sort=False)

        differences = configs.T.apply(lambda x: not all(i == x[0] for i in x),
                                      axis=1)

        tmp = configs.T[differences].T
        pd.set_option('display.max_columns', 500)
        tmp = tmp.sort_values('run')

        return tmp

    def gif_wall(self, runs=None, target='rollout_00.gif', file='gif_wall.html'):
        """Create a html file, which displays the target gifs for all runs."""
        if runs is None:
            runs = list(self.runs)
        gif_path = os.path.join('gifs', target)

        gif_html = """
            <div class="img-frame-cap">
            <img src="{}" width="180" height="180+15">
            <div class="caption"> {} </div>
            </div>
            """

        style = """
            <style>
            body {
                background-color: #000000;
            }
            .img-frame-cap {
                width:200px;
                height:200px;
                background:#333;
                padding:18px 18px 2px 18px;
                border:1px solid #666;
                display:inline-block;
            }

            .caption {
                text-align:center;
                margin-top:4px;
                font-size:12px;
                color: #FFFFFF;
            }

            </style>
            """
        skeleton = [
            '<HTML>', '<HEAD> <TITLE>Gif WALL</TITLE> </HEAD>',
            style,
            '<BODY>', '</BODY>', '</HTML>']

        skeleton = [i+'\n' for i in skeleton]

        with open(os.path.join('test', file), 'w') as f:
            f.write(skeleton[0])
            f.write(skeleton[1])
            f.write(style)
            for run in runs:
                location = os.path.join('../', run, gif_path)
                f.write(gif_html.format(location, location))
            f.write(skeleton[-2])
            f.write(skeleton[-1])

        # print url for easy access
        print(
            'file:///Users/{}/Documents/remote/thesis/code/'.format(self.user)
            + '/'.join(os.getcwd().split('/')[-3:])
            + '/test/gif_wall.html')

    def perf_plot(self, perf, z, ax, col='error', label=None, rol=1, color=None):
        """For a given run and z, plot performance column.

        Note that perf=self.perfs[run].
        """
        tmp = perf.loc[perf.type == z, col]
        if 'roll' in z:
            tmp = tmp.groupby('step').mean()
        if label is None:
            label = z
        tmp.rolling(rol).mean().plot(label=label, alpha=0.5, ax=ax, c=color)

        return tmp

    def plot_error(self, run, perf, col, rol=1):
        """For a given run and col, plot error for all zs."""
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.title(run + " " + self.confs[run].T.description.values[0])

        for z in self.zs:
            try:
                self.perf_plot(perf, z, plt.gca(), col, rol=rol)
            except Exception as e:
                print(run, e)
        plt.legend()

        return fig, ax

    def compare_errors(self, runs=None, rol=1, col='error', colors=None):
        """For a given col, plot errors for all runs and zs."""
        if runs is None:
            runs = self.runs
        fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        for run in runs:
            if colors is not None:
                c = colors[1][run]
            else:
                c = None

            perf = self.perfs[run]
            for z, ax in zip(self.zs, axs):
                try:
                    self.perf_plot(perf, z, ax, col=col, label=run, rol=rol, color=c)
                except Exception as e:
                    print(run, e)

        axs[1].legend(bbox_to_anchor=(1.05, 1.05))

        if colors is not None:
            for i, (value, color) in enumerate(colors[0].items()):
                axs[-1].text(
                    0.9, 0.9-0.1*i, str(value), horizontalalignment='right',
                    verticalalignment='top', transform=axs[1].transAxes, color=color)

        return fig, axs

    def speed_comparison(self, runs=None):
        """Compare speeds of runs."""
        if runs is None:
            runs = self.runs

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        for run in runs:
            perf = self.perfs[run]

            if not perf.size > 0:
                continue
            try:
                perf.loc[perf.type == 'z'].time.plot(label=run, alpha=.5, ax=ax)

            except Exception as e:
                print(e)
                print('Not possible for run {}'.format(run))
                perf.time.plot(label=run, alpha=0.5, ax=ax)

        plt.legend(bbox_to_anchor=(1.05, 1.05))
        plt.show()

        return fig, ax

    def compare_runs(self, col='elbo', description=None, legend=False,
                     runs=None, colors=None, **kwargs):
        """For a given col, compare values over all plots.

        Unlike previous methods, this is only used for metric, which do not
        differ between zs, such as ['elbo', 'bg', 'patch',...]
        """
        fig, ax = plt.subplots(1, figsize=(12, 6))
        plt.title(col)

        if runs is None:
            runs = self.runs
        for run in runs:
            perf = self.perfs[run]

            # only plot if description matches
            if description is not None:
                if not all(self.confs[run].T.description == description):
                    continue
            # ignore empty runs
            try:
                # if multiple types present, select any but rollout
                if 'type' in perf.columns:
                    # choose any type but rollout
                    z = list(set(perf.type.unique()) - {'rollout'})
                    z = [i for i in z if 'roll' not in i]
                    z = z[0]
                    df_plot = perf[perf.type == z]
                else:
                    df_plot = perf
                if colors is not None:
                    c = colors[1][run]
                else:
                    c = None

                df_plot.plot(y=col, ax=ax, label=run, alpha=0.9, legend=legend,
                             color=c, **kwargs)

                if legend is True:
                    plt.legend(bbox_to_anchor=(1.05, 1.05))

            except Exception as e:
                print(e)
                print(run)

        if colors is not None:
            for i, (value, color) in enumerate(colors[0].items()):
                ax.text(
                    0.9, 0.9-0.1*i, str(value), horizontalalignment='right',
                    verticalalignment='top', transform=ax.transAxes, color=color)

        return fig, ax

    def get_highlighting(self, highlighter, condition=None, runs=None):
        """Color runs depending on config.

        All methods with color argument take color_map returned by this function
        to color runs depending on their value on one of the configs arguments.

        Example: For highlighter='num_epochs'. All runs are colored by their
        number of epochs: same color for same num_epochs, each distinct
        num_epoch value is assigned a different color.

        Returns:
            color_map (dict): Contains mapping of unique highlighter values
                to colors.
            mapping (dict): Contains mapping of runs to colors, depending on
                value highlighter takes in conf.
            Pass colors as argument to other functions.

        """
        if runs is None:
            runs = self.runs

        # first, apply condition to all runs
        possible_values = set()
        realised_values = dict()

        for run in runs:
            try:
                conf = self.confs[run].T[highlighter].values[0]
            except Exception as e:
                print(e)
                print("Did not find conf '{}' for '{}', set to None.".format(
                    highlighter, run))
                conf = None

            if condition is not None:
                # check more than just content of column
                conf = condition(conf)

            possible_values.update({conf})
            realised_values.update({run: conf})

        color_map = dict(zip(list(possible_values), self.color_cylce))

        mapping = {r: color_map[v] for r, v in realised_values.items()}

        colors = [color_map, mapping]
        return colors

    def aggregate_comparisons(self, description=None, with_z=False):
        """Plot compare_runs for common metrics."""
        compare_runs = self.compare_runs
        try:
            compare_runs('elbo', legend=True, description=description)
            compare_runs('bg', description=description)
            compare_runs('patch', description=description)
            compare_runs('overlap', description=description)
            compare_runs('log_q', description=description)
            if with_z:
                compare_runs('scale_x', description=description)
                compare_runs('scale_y', description=description)
                compare_runs('error', description=description)
                compare_runs('std_error', description=description)
        except Exception as e:
            print(e)

    def plot_single_rollout(self, run, col='x_errors'):
        """Plot rollout error evolution for a single run."""
        if col is None:
            col = 'x_errors'

        # always load x_error (see rollout property)
        error = self.rollouts[col][run]

        fig, ax = plt.subplots(1, 1)

        error.T.plot(label='', legend=None, color='gray', alpha=0.1, ax=ax)
        error.mean(axis=0).plot(ax=ax)

        # avg error over first and last 10 epochs
        error.iloc[-5:-1].mean(axis=0).plot(c='g')
        error.iloc[0:5].mean(axis=0).plot(c='r')

        # start of inference
        # inference = int(self.confs[run].T.num_visible.values[0])\
        #      - int(self.confs[run].T.skip.values[0])
        # plt.plot([inference, inference], [0, 0.3])

        ax.set_title(run)

        return fig, ax

    def plot_all_rollouts(self, runs=None, col=None):
        """Plot rollout error evolutions for all runs."""
        if runs is None:
            runs = self.runs

        figs, axs = [], []

        for run in list(self.runs):
            try:
                f, a = self.plot_single_rollout(run, col)
                figs.append(f)
                axs.append(a)

            except Exception as e:
                print(e)
                print('Not available for run {}'.format(run))

        return figs, axs

    def plot_compare_rollouts(self, runs=None, which='x_errors'):
        """Plot final rollout errors for all runs."""
        if runs is None:
            runs = self.runs

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        errors = self.rollouts[which]

        for run in list(runs):
            try:
                error = errors[run]
                error.iloc[-1].plot(label=run)
            except Exception as e:
                print(e)
                print('Not available for run {}'.format(run))

        ax.legend(bbox_to_anchor=(1.05, 1.05))
        ax.set_title('avg over last 5 epochs')

        return fig, ax

    @property
    def rollouts(self):
        """Load rollout errors as property."""
        # check if already loaded
        if self._rollouts is not None:
            return self._rollouts

        # if not present, load rollout errors from csv
        v_errors = {}
        x_errors = {}
        v_errors_std = {}
        x_errors_std = {}

        x_errors_sup = {}
        x_errors_std_sup = {}

        for run in list(self.runs):
            try:
                x_errors[run] = pd.read_csv(os.path.join(run, 'error.csv'), header=None)
                x_errors_std[run] = pd.read_csv(os.path.join(run, 'std_error.csv'), header=None)
            except Exception as e:
                x_errors[run] = None
                x_errors_std[run] = None
                print(e)
                print('x_error available for run {}'.format(run))
            try:
                v_errors[run] = pd.read_csv(os.path.join(run, 'v_error.csv'), header=None)
                v_errors_std[run] = pd.read_csv(os.path.join(run, 'std_v_error.csv'), header=None)
            except Exception as e:
                v_errors[run] = None
                v_errors_std[run] = None
                print(e)
                print('v_error not available for run {}'.format(run))

            # Load supervised errors if model was supervised.
            # If error or not supervised write None.
            try:
                if self.confs[run].T.supairvised.values[0] != 'True':
                    raise
                x_errors_sup[run] = pd.read_csv(os.path.join(run, 'error_sup.csv'), header=None)
                x_errors_std_sup[run] = pd.read_csv(os.path.join(run, 'std_error_sup.csv'), header=None)
            except Exception as _:
                x_errors_sup[run] = None
                x_errors_std_sup[run] = None

            try:
                if self.confs[run].T.supairvised.values[0] != 'True':
                    raise
                x_errors_sup[run] = pd.read_csv(os.path.join(run, 'error_sup.csv'), header=None)
                x_errors_std_sup[run] = pd.read_csv(os.path.join(run, 'std_error_sup.csv'), header=None)

            except Exception as _:
                x_errors_sup[run] = None
                x_errors_std_sup[run] = None

        self._rollouts = dict()
        self._rollouts['v_errors'] = v_errors
        self._rollouts['x_errors'] = x_errors
        self._rollouts['v_errors_std'] = v_errors_std
        self._rollouts['x_errors_std'] = x_errors_std

        self._rollouts['x_errors_sup'] = x_errors_sup
        self._rollouts['x_errors_std_sup'] = x_errors_std_sup

        return self._rollouts

    @staticmethod
    def run_fmt(x, with_under=False):
        """Format list of ints as run-strings."""
        return 'run{:03d}'.format(x) if not with_under else 'run_{:03d}'.format(x)

    @staticmethod
    def descr2folder(run):
        """Find a run given its config description.

        If multiple matches, returns first only.
        """
        for folder_run, conf in self.confs.items():
            if conf.T.description.values[0][-3:] == run[-3:]:
                return folder_run

        print('Run {} cant be found'.format(run))
        return -1

    @staticmethod
    def folder2descr(run):
        """Given a run, return config description."""
        return self.confs[run].T.description.values[0]

    def plot_supairvised_errors(self, runs=None):
        """Plot errors for supairvised model.

        Some info on the displayed columns for to remember:

        - vin_true pred False/True is the most interesting error.
            Prediction error not always measured against true labels, for
            use_supair=True this is measured as model against supair values.
        - vin_sup pred False/True is also interesting. How good is vin
            prediction w.r.t supair values.

        (There should be no big differences for False/True (train/test) here.
        If there are, we are overfitting on the training set.)

        - sup_true recon/pred/total false/true should all be roughly the same
            b/c supair is pretrained and should perform the same no matter
            on which images it is applied.

        - vin_sup recon is always 0, bc vin does not encode supair states
        - vin_true recon is equal to sup_true recon bc of that

        """
        if runs is None:
            runs = self.runs

        groups_dict = dict(zip(self.supairvised_groups, range(12)))

        fig, axs = plt.subplots(12, 1, figsize=(15, 25), sharex=True)
        for name, i in groups_dict.items():
            axs[i].set_title(name)

        for run in runs:
            perf = self.perfs[run]
            # automatically skip all non supairvised runs
            try:
                supairvised = bool(self.confs[run].T.supairvised.values[0])
                if not supairvised:
                    continue
            except Exception as _:
                continue

            groups = perf.groupby(['type', 'frame', 'test'])
            for name, group in groups:
                i = groups_dict[name]
                group.error.plot(label=run, ax=axs[i])

        axs[0].legend(bbox_to_anchor=(1.05, 1.05))
        plt.tight_layout()

        return fig, axs

    def plot_relevant_supairvised_errors(self, runs=None):
        """Plot only relevant errors for supairvised models.

        A lot of the information from plot_supairvised_errors is redundant.
        Only plot the most relevant columns to ease confusion.
        """
        if runs is None:
            runs = self.runs

        relevant_cols = [
            ('vin_true', 'pred', True),
            ('vin_true', 'pred', False),
            ('vin_sup', 'pred', False),
        ]

        groups_dict = dict(zip(relevant_cols, range(12)))

        fig, axs = plt.subplots(len(relevant_cols), 1, figsize=(15, 15), sharex=True)
        for name, i in groups_dict.items():
            axs[i].set_title(name)

        for run in runs:
            perf = self.perfs[run]
            # automatically skip all non supairvised runs
            try:
                supairvised = bool(self.confs[run].T.supairvised.values[0])
                if not supairvised:
                    continue
            except Exception as e:
                continue

            groups = perf.groupby(['type', 'frame', 'test'])
            for name, group in groups:
                try:
                    i = groups_dict[name]
                    group.error.plot(label=run, ax=axs[i])
                except Exception as e:
                    continue

        axs[1].legend(bbox_to_anchor=(1.05, 1.05))
        axs[0].legend(bbox_to_anchor=(1.05, 1.05))
        plt.tight_layout()

        return fig, axs

    def compare_rollouts_with_supairvised(self, runs=None):
        """Add suparivised models to rollout plots."""
        if runs is None:
            runs = self.runs

        fig, ax = plt.subplots(figsize=(16, 10))

        for run in runs:
            perf = self.perfs[run]

            # figure out if supairvised or not
            supairvised = bool(self.confs[run].T.supairvised.values[0])

            if supairvised:
                cond = (
                    (perf.type == 'vin_true') &
                    (perf.frame == 'pred') &
                    (perf.test == True))
                perf = perf.loc[cond]
            else:
                perf = perf.loc[perf.type == 'rollout']

            perf.groupby('step').error.mean().plot(label=run, ax=ax)

        ax.legend(bbox_to_anchor=(1.05, 1.05))
        plt.tight_layout()
        plt.show()

        return fig, ax
