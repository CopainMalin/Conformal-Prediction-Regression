# Adaptation of the code that you can find here :
# https://mapie.readthedocs.io/en/latest/examples_regression/4-tutorials/plot_cqr_tutorial.html
# To the following dataset :
# https://www.kaggle.com/datasets/mirichoi0218/insurance
# With some little changes (estimator choice using sklearn pipelines, etc)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aquarel import load_theme
from matplotlib.offsetbox import AnnotationBbox, TextArea
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeRegressor
from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score, regression_mean_width_score
import warnings

warnings.filterwarnings("ignore")

ALPHA = 0.2  # Niveau de couverture visé (1 - alpha donc 80% ici)

STRATEGIES = {
    "naive": {"method": "naive"},
    "cv_plus": {"method": "plus", "cv": 10},
    "jackknife_plus": {"method": "plus", "cv": -1},
    "base": {"method": "base", "cv": "split"},
}

NAMES = {
    "naive": "Naive",
    "cv_plus": "CV+",
    "jackknife_plus": "Jacknife+",
    "base": "Inductive CP",
}

## graphics theme ##
theme = load_theme("arctic_dark")
theme.apply()


## functions  ##
# https://mapie.readthedocs.io/en/latest/examples_regression/4-tutorials/plot_cqr_tutorial.html
def sort_y_values(y_test, y_pred, y_pis):
    """
    Sorting the dataset in order to make plots using the fill_between function.
    """
    indices = np.argsort(y_test)
    y_test_sorted = np.array(y_test)[indices]
    y_pred_sorted = y_pred[indices]
    y_lower_bound = y_pis[:, 0, 0][indices]
    y_upper_bound = y_pis[:, 1, 0][indices]
    return y_test_sorted, y_pred_sorted, y_lower_bound, y_upper_bound


def plot_prediction_intervals(
    title,
    axs,
    y_test_sorted,
    y_pred_sorted,
    lower_bound,
    upper_bound,
    coverage,
    width,
    num_plots_idx,
):
    """
    Plot of the prediction intervals for each different conformal
    method.
    """
    axs.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    axs.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    lower_bound_ = np.take(lower_bound, num_plots_idx)
    y_pred_sorted_ = np.take(y_pred_sorted, num_plots_idx)
    y_test_sorted_ = np.take(y_test_sorted, num_plots_idx)

    error = y_pred_sorted_ - lower_bound_

    warning1 = y_test_sorted_ > y_pred_sorted_ + error
    warning2 = y_test_sorted_ < y_pred_sorted_ - error
    warnings = warning1 + warning2
    axs.errorbar(
        y_test_sorted_[~warnings],
        y_pred_sorted_[~warnings],
        yerr=np.abs(error[~warnings]),
        color="#29bf12",
        capsize=5,
        marker="o",
        elinewidth=2,
        linewidth=0,
        label="Inside prediction interval",
    )
    axs.errorbar(
        y_test_sorted_[warnings],
        y_pred_sorted_[warnings],
        yerr=np.abs(error[warnings]),
        capsize=5,
        marker="o",
        elinewidth=2,
        linewidth=0,
        color="red",
        label="Outside prediction interval",
    )
    axs.scatter(
        y_test_sorted_[warnings],
        y_test_sorted_[warnings],
        marker="*",
        color="dodgerblue",
        label="True value",
    )
    axs.set_xlabel("True charges in $")
    axs.set_ylabel("Charges prediction in $")
    ab = AnnotationBbox(
        TextArea(
            f"Coverage: {np.round(coverage, 3)}\n"
            + f"Interval width: {np.round(width, 3)}\n"
        ),
        bboxprops={"facecolor": "#687691", "edgecolor": "white"},
        xy=(np.min(y_test_sorted_) * 3, np.max(y_pred_sorted_ + error) * 0.95),
    )
    lims = [
        np.min([axs.get_xlim(), axs.get_ylim()]),  # min of both axes
        np.max([axs.get_xlim(), axs.get_ylim()]),  # max of both axes
    ]
    axs.plot(lims, lims, "--", alpha=0.75, color="black", label="x=y")
    axs.add_artist(ab)
    axs.set_title(title, fontweight="bold")


## Data loading ##
# https://www.kaggle.com/datasets/mirichoi0218/insurance

if __name__ == "__main__":
    insurance = pd.read_csv("datas/insurance.csv")
    X = insurance.iloc[:, :-1]
    y = insurance.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    ## Preprocessing ##
    CAT_COLUMNS = X.select_dtypes(exclude=np.number).columns
    NUMERIC_COLUMNS = X.select_dtypes(include=np.number).columns

    numerical_pipeline = make_pipeline(StandardScaler())
    categorical_pipeline = make_pipeline(OneHotEncoder())

    preprocessor = make_column_transformer(
        (numerical_pipeline, NUMERIC_COLUMNS), (categorical_pipeline, CAT_COLUMNS)
    )

    y_pred, y_pis = {}, {}
    y_test_sorted, y_pred_sorted, lower_bound, upper_bound = {}, {}, {}, {}
    coverage, width = {}, {}
    for strategy, params in STRATEGIES.items():
        estimator = make_pipeline(preprocessor, DecisionTreeRegressor())
        mapie = MapieRegressor(estimator, **params, random_state=0)
        mapie.fit(X_train, y_train)
        y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=ALPHA)
        (
            y_test_sorted[strategy],
            y_pred_sorted[strategy],
            lower_bound[strategy],
            upper_bound[strategy],
        ) = sort_y_values(y_test, y_pred[strategy], y_pis[strategy])
        coverage[strategy] = regression_coverage_score(
            y_test, y_pis[strategy][:, 0, 0], y_pis[strategy][:, 1, 0]
        )
        width[strategy] = regression_mean_width_score(
            y_pis[strategy][:, 0, 0], y_pis[strategy][:, 1, 0]
        )

    perc_obs_plot = 0.02
    num_plots = np.arange(len(y_test))
    fig, axs = plt.subplots(2, 2, figsize=(15, 13))
    coords = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

    for strategy, coord in zip(STRATEGIES.keys(), coords):
        plot_prediction_intervals(
            NAMES[strategy]
            + f" | Decision Tree | Targeted coverage : {(1-ALPHA)*100:.0f}%",
            coord,
            y_test_sorted[strategy],
            y_pred_sorted[strategy],
            lower_bound[strategy],
            upper_bound[strategy],
            coverage[strategy],
            width[strategy],
            num_plots,
        )

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(_, []) for _ in zip(*lines_labels)]
    plt.legend(
        lines[:4],
        labels[:4],
        loc="upper center",
        bbox_to_anchor=(0, -0.15),
        fancybox=True,
        shadow=True,
        ncol=2,
    )
    plt.savefig("plots/insurance_ci.svg", bbox_inches="tight")
