import pandas as pd
from matplotlib.pyplot import figure
from xailib.models.bbox import AbstractBBox
from xailib.xailib_tabular import TabularExplainer, TabularExplanation
import shap
import matplotlib.pyplot as plt
shap.initjs

#to chart with altair
import altair as alt
from altair import expr
import numpy as np

from IPython.display import HTML      


class ShapXAITabularExplanation(TabularExplanation):
    def __init__(self, shap_exp, feature_names: list):
        super().__init__()
        self.exp = shap_exp
        self.feature_names = feature_names

    def getFeaturesImportance(self):
        return self.exp

    def getExemplars(self):
        return None

    def getCounterExemplars(self):
        return None

    def getRules(self):
        return None

    def getCounterfactualRules(self):
        return None

    def plot_features_importance(self, fontDimension=10):
        #data prepraration
        feature_values = []
        if np.ndim(self.exp) == 1 :
            feature_values = self.exp
        else:
            feature_values = self.exp[0]


        arr=np.c_[np.array(self.feature_names), feature_values]
        dataToPlot=pd.DataFrame(arr,columns=['name','value'])
        dataToPlot['value'] = dataToPlot['value'].astype('float64')

        fontSize = fontDimension
        step = fontSize*1.5

        maxValue = dataToPlot['value'].max()
        minValue = dataToPlot['value'].min()
        maxRange = max(abs(maxValue),abs(minValue))
        
        #selector
        slider = alt.binding_range(min=0, max=maxRange, step=maxRange/50, name='Importance cutoff value (±) ')
        selector = alt.selection_single(name="Cutter", fields=['cutoff'], bind=slider, init={'cutoff': 0.0})

        #charting
        bar= alt.Chart(
            dataToPlot
        ).transform_filter(
             (alt.datum.value > selector.cutoff ) | (alt.datum.value < -(selector.cutoff))
        ).mark_bar().encode(
            x=alt.X('value:Q',title=None),
            y=alt.Y('name:N',title=None, sort=alt.EncodingSortField(field='value', op='mean',order='descending')),
            color=alt.Color(
              'value:Q',
               scale=alt.Scale(
                    scheme='blueorange',
                    domain=[-maxRange,maxRange],
                    domainMid=0,
                    ),
              legend=None
                ),
            tooltip=[
                 alt.Tooltip(field='name',type='nominal', title='Feature'),
                 alt.Tooltip(field='value',type='quantitative', title='Importance')
                             ]
        ).add_selection(
        selector
        )
        line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule().encode(x='x')

        #Legend Chart
        legendData=np.arange(-maxRange, maxRange, maxRange/100).tolist()
        legendDF=pd.DataFrame({'xValue': legendData})

        legendChart = alt.Chart(
            legendDF
        ).mark_rule(
        strokeWidth=3
        ).encode(
            x=alt.X(
                field='xValue',
                type='quantitative',
                title='Select a cutoff range for Feature Importance values ',
                axis=alt.Axis(orient='top',titleFontSize=fontSize)
            ),
            color=  alt.Color(
              'xValue:Q',
               scale=alt.Scale(
                    scheme='redyellowblue',
                    domain=[-maxRange, maxRange],
                    domainMid=0,
                    ),
                legend=None
            )
        )
        
        cuttedChart= alt.Chart(
            pd.DataFrame({'y': [0],'x': [-0.5],'x2': [0.5]})
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
        
        legend=(legendChart+cuttedChart).properties(
            height=20
        )
        
        chart=(legend & (bar+line)).properties(
            padding=10,
            ).configure_axis(
            labelLimit=step*15,
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


class NoExplainerFound(Exception):

    def __init__(self, name):
        self.message = 'Explanator not found '+name
        super().__init__(self.message)


class ShapXAITabularExplainer(TabularExplainer):
    shap_explainer = None

    def __init__(self, bb: AbstractBBox, feature_names: list):
        super().__init__()
        self.bb = bb
        self.feature_names = feature_names

    def fit(self, config):
        if config['explainer'] == 'linear':
            self.shap_explainer = shap.LinearExplainer(self.bb.model(), config['X_train'])
        elif config['explainer'] == 'tree':
            self.shap_explainer = shap.TreeExplainer(self.bb.model())
        elif config['explainer'] == 'deep':
            self.shap_explainer = shap.DeepExplainer(self.bb, config['X_train'])
        elif config['explainer'] == 'kernel':
            self.shap_explainer = shap.KernelExplainer(self.bb.predict_proba, config['X_train'])
        else:
            raise NoExplainerFound(config['explainer'])



    def explain(self, x):
        exp = self.shap_explainer.shap_values(x)
        return ShapXAITabularExplanation(exp, self.feature_names)

    def expected_value(self, val):
        if val == -1:
            return self.shap_explainer.expected_value
        else:
            return self.shap_explainer.expected_value[val]

    def plot_shap_values(self, feature_names, exp, range_start, range_end):
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(10, 8))
        plt.bar(feature_names[range_start:range_end], exp[range_start:range_end], facecolor='lightblue', width=0.5)
        # You can specify a rotation for the tick labels in degrees or with keywords.
        plt.xticks(feature_names[range_start:range_end], rotation='vertical')
        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.1)
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.25)
        plt.show()

    