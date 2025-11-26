"""
Tabular data explainability classes for XAI-Lib.

This module provides base classes for explaining predictions on tabular
(structured) data. It extends the base :class:`~xailib.xailib_base.Explainer`
and :class:`~xailib.xailib_base.Explanation` classes with tabular-specific
functionality, including interactive feature importance visualization.

Tabular data explanations are commonly used for:
    - Understanding feature contributions to predictions
    - Generating human-readable decision rules
    - Identifying similar and contrasting examples
    - Creating counterfactual explanations

Classes:
    TabularExplanation: Base class for tabular data explanations.
    TabularExplainer: Base class for tabular data explainers.

Example:
    Using LIME for tabular explanation::

        from xailib.explainers.lime_explainer import LimeXAITabularExplainer
        from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper

        # Wrap your model
        bb = sklearn_classifier_wrapper(your_sklearn_model)

        # Create and fit explainer
        explainer = LimeXAITabularExplainer(bb)
        explainer.fit(df, 'target_column', config={})

        # Generate explanation
        explanation = explainer.explain(instance)
        explanation.plot_features_importance()

See Also:
    :mod:`xailib.explainers.lime_explainer`: LIME implementation for tabular data.
    :mod:`xailib.explainers.shap_explainer_tab`: SHAP implementation for tabular data.
    :mod:`xailib.explainers.lore_explainer`: LORE implementation for tabular data.
"""

from abc import abstractmethod
from xailib.xailib_base import Explainer, Explanation
import pandas as pd
import numpy as np

import altair as alt
from altair import expr
from IPython.display import HTML


class TabularExplanation(Explanation):
    """
    Abstract base class for tabular data explanations.

    This class extends the base :class:`~xailib.xailib_base.Explanation` class
    with functionality specific to tabular (structured) data, including
    interactive visualization of feature importance using Altair charts.

    Subclasses should implement the abstract methods to provide access
    to different types of explanation information (feature importance,
    rules, exemplars, etc.).

    Attributes:
        Defined by subclasses. Common attributes include the raw explanation
        object from the underlying library.

    See Also:
        :class:`xailib.explainers.lime_explainer.LimeXAITabularExplanation`: LIME explanation.
        :class:`xailib.explainers.shap_explainer_tab.ShapXAITabularExplanation`: SHAP explanation.
        :class:`xailib.explainers.lore_explainer.LoreTabularExplanation`: LORE explanation.
    """

    def __init__(self):
        """Initialize the TabularExplanation base class."""
        super().__init__()

    @abstractmethod
    def getFeaturesImportance(self):
        """
        Get feature importance values for the explained instance.

        Returns:
            Feature importance as a list of tuples, numpy array, or
            pandas DataFrame. The exact format depends on the explanation
            method. Returns None if feature importance is not available.
        """
        pass

    @abstractmethod
    def getExemplars(self):
        """
        Get exemplar instances similar to the explained instance.

        Returns:
            Exemplar instances with the same prediction, or None if
            not supported by this explanation method.
        """
        pass

    @abstractmethod
    def getCounterExemplars(self):
        """
        Get counter-exemplar instances with different predictions.

        Returns:
            Counter-exemplar instances, or None if not supported
            by this explanation method.
        """
        pass

    @abstractmethod
    def getRules(self):
        """
        Get decision rules explaining the prediction.

        Returns:
            Decision rules as a dictionary or list, or None if not
            supported by this explanation method.
        """
        pass

    @abstractmethod
    def getCounterfactualRules(self):
        """
        Get counterfactual rules for alternative outcomes.

        Returns:
            Counterfactual rules describing how to change the prediction,
            or None if not supported by this explanation method.
        """
        pass

    def plot_features_importance_from(self, dataToPlot: pd.DataFrame, fontDimension=10):
        """
        Create an interactive feature importance visualization using Altair.

        This method generates an interactive bar chart showing feature importance
        values with a slider to filter features by importance threshold. Features
        are color-coded by their importance value (positive vs negative).

        Args:
            dataToPlot (pd.DataFrame): DataFrame containing feature importance data
                with columns:
                    - 'name': Feature names (string)
                    - 'value': Importance values (float)
            fontDimension (int, optional): Base font size for the chart. Defaults to 10.

        Returns:
            None. Displays the interactive chart using IPython display.

        Note:
            This method is intended to be called within a Jupyter notebook
            environment for proper rendering of the interactive chart.

        Example:
            >>> import pandas as pd
            >>> data = pd.DataFrame({
            ...     'name': ['feature1', 'feature2', 'feature3'],
            ...     'value': [0.5, -0.3, 0.1]
            ... })
            >>> explanation.plot_features_importance_from(data, fontDimension=12)
        """
        fontSize = fontDimension
        step = fontSize * 1.5

        maxValue = dataToPlot['value'].max()
        minValue = dataToPlot['value'].min()
        maxRange = max(abs(maxValue), abs(minValue))

        # selector
        slider = alt.binding_range(min=0, max=maxRange, step=maxRange / 50, name='Importance cutoff value (±) ')
        selector = alt.selection_single(name="Cutter", fields=['cutoff'], bind=slider, init={'cutoff': 0.0})

        # charting
        bar = alt.Chart(
            dataToPlot
        ).transform_filter(
            (alt.datum.value > selector.cutoff) | (alt.datum.value < -(selector.cutoff))
        ).mark_bar().encode(
            x=alt.X('value:Q', title=None),
            y=alt.Y('name:N', title=None, sort=alt.EncodingSortField(field='value', op='mean', order='descending')),
            color=alt.Color(
                'value:Q',
                scale=alt.Scale(
                    scheme='blueorange',
                    domain=[-maxRange, maxRange],
                    domainMid=0,
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip(field='name', type='nominal', title='Feature'),
                alt.Tooltip(field='value', type='quantitative', title='Importance')
            ]
        ).add_selection(
            selector
        )
        line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule().encode(x='x')

        # Legend Chart
        legendData = np.arange(-maxRange, maxRange, maxRange / 100).tolist()
        legendDF = pd.DataFrame({'xValue': legendData})

        legendChart = alt.Chart(
            legendDF
        ).mark_rule(
            strokeWidth=3
        ).encode(
            x=alt.X(
                field='xValue',
                type='quantitative',
                title='Select a cutoff range for Feature Importance values ',
                axis=alt.Axis(orient='top', titleFontSize=fontSize)
            ),
            color=alt.Color(
                'xValue:Q',
                scale=alt.Scale(
                    scheme='redyellowblue',
                    domain=[-maxRange, maxRange],
                    domainMid=0,
                ),
                legend=None
            )
        )

        cuttedChart = alt.Chart(
            pd.DataFrame({'y': [0], 'x': [-0.5], 'x2': [0.5]})
        ).transform_calculate(
            x_min=-(selector.cutoff),
            x_max=selector.cutoff
        ).mark_rect(
            color='black',
            height=20,
            tooltip=True,
            opacity=0.4
        ).encode(
            x='x_min:Q',
            x2='x_max:Q',
            y=alt.Y(field="y", type="quantitative", axis=None)

        ).add_selection(
            selector
        )

        legend = (legendChart + cuttedChart).properties(
            height=20
        )

        chart = (legend & (bar + line)).properties(
            padding=10,
        ).configure_axis(
            labelLimit=step * 15,
            labelFontSize=fontSize
        )

        # HTML injection using IPython.display  
        display(HTML("""
        <style>
        .vega-bind {
          position: absolute;
          left: 0px;
          top: 0px;
          background-color:#eee;
          padding:10px;
          font-size:%spx;
        }
        .chart-wrapper{
          padding-top: 70px;
        }

        </style>
        """ % (fontSize)
                     ))
        display(chart)


class TabularExplainer(Explainer):
    """
    Abstract base class for tabular data explainers.

    This class extends the base :class:`~xailib.xailib_base.Explainer` class
    with functionality specific to tabular (structured) data. Tabular
    explainers work with pandas DataFrames and provide explanations
    for predictions on structured data.

    Subclasses implement specific explanation methods such as LIME, SHAP,
    or LORE for tabular data.

    Attributes:
        Defined by subclasses. Common attributes include the black-box model
        wrapper and configuration parameters.

    See Also:
        :class:`xailib.explainers.lime_explainer.LimeXAITabularExplainer`: LIME implementation.
        :class:`xailib.explainers.shap_explainer_tab.ShapXAITabularExplainer`: SHAP implementation.
        :class:`xailib.explainers.lore_explainer.LoreTabularExplainer`: LORE implementation.
    """

    def __init__(self):
        """Initialize the TabularExplainer base class."""
        super().__init__()

    @abstractmethod
    def fit(self, X, y, config):
        """
        Fit the explainer to the tabular training data.

        Args:
            X (pd.DataFrame): Training data as a pandas DataFrame.
            y: Target column name (str) or target values.
            config (dict): Configuration dictionary with method-specific parameters.
                Common keys include:
                    - 'feature_selection': Feature selection method
                    - 'discretize_continuous': Whether to discretize continuous features
                    - 'sample_around_instance': Sampling strategy
                    - Additional method-specific parameters

        Returns:
            None. The explainer is fitted in-place.
        """
        pass

    @abstractmethod
    def explain(self, b, x) -> TabularExplanation:
        """
        Generate an explanation for a tabular data instance.

        Args:
            b: Black-box model or prediction function (depends on implementation).
            x: Instance to explain as a numpy array or pandas Series.

        Returns:
            TabularExplanation: An explanation object containing feature importance,
                rules, or other explanation information.
        """
        pass
