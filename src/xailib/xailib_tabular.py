from abc import abstractmethod
from xailib.xailib_base import Explainer, Explanation
import pandas as pd
import numpy as np

import altair as alt
from altair import expr
from IPython.display import HTML


class TabularExplanation(Explanation):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def getFeaturesImportance(self):
        pass

    @abstractmethod
    def getExemplars(self):
        pass

    @abstractmethod
    def getCounterExemplars(self):
        pass

    @abstractmethod
    def getRules(self):
        pass

    @abstractmethod
    def getCounterfactualRules(self):
        pass

    def plot_features_importance_from(self, dataToPlot: pd.DataFrame, fontDimension=10):
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

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y, config):
        pass

    @abstractmethod
    def explain(self, b, x) -> TabularExplanation:
        pass
