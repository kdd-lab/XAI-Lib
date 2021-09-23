import pandas as pd
from matplotlib.pyplot import figure
from xailib.models.bbox import AbstractBBox
from xailib.xailib_tabular import TabularExplainer
import shap
import matplotlib.pyplot as plt
shap.initjs

#to chart with altair
import altair as alt
from altair import expr
import numpy as np

from IPython.display import HTML      


class NoExplainerFound(Exception):

    def __init__(self, name):
        self.message = 'Explanator not found '+name
        super().__init__(self.message)


class ShapXAITabularExplainer(TabularExplainer):
    shap_explainer = None

    def __init__(self, bb: AbstractBBox):
        super().__init__()
        self.bb = bb

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
        return exp

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

    def plot_shap_values_alt(self, feature_names, exp, fontDimension=10):
    
        fontSize = fontDimension
    
        #data prepraration
        arr=np.c_[np.array(feature_names),exp]
        dataToPlot=pd.DataFrame(arr,columns=['name','value'])

        
        #chart selector
        dotData=np.arange(-1, 1, 0.0125).tolist()
        dotDF=pd.DataFrame({'xValue': dotData})
                           
        brush = alt.selection_interval(
            encodings=['x'],
            init = {"xValue":[-0.005,0.005]},
            fields= ['xValue'],
            mark=alt.BrushConfig(fill='green'),
            name='sel')

        dotSelection = alt.Chart(
            dotDF
        ).mark_rule(
        strokeWidth=3
        ).encode(
            x=alt.X(
                field='xValue',
                type='quantitative',
                title='Select a cutoff range for Feature Importance values ',
                axis=alt.Axis(orient='top',titleFontSize=fontSize)
            ),
            color=alt.condition(
                'datum.xValue < sel.xValue[0] | datum.xValue > sel.xValue[1]',
                alt.Color(
              'xValue:Q',
               scale=alt.Scale(
                    scheme='blueorange',
                    domain=[-1,1],
                    domainMid=0,
                    ),
                    legend=None
            ),
                alt.value('lightgray'))
        ).add_selection(
        brush
        )


        #charting bars
        bars= alt.Chart(
            dataToPlot
        ).transform_filter(
            'datum.value > -1 & datum.value < 1'
        ).transform_calculate(
            minSelection="toNumber(sel.xValue[0])",
            maxSelection="toNumber(sel.xValue[1])"
        ).transform_filter(
             'datum.value < sel.xValue[0] | datum.value > sel.xValue[1]'
        ).mark_bar().encode(
            x=alt.X('value:Q',title=None),
            y=alt.Y('name:N',title=None, sort=alt.EncodingSortField(field='value', op='mean',order='descending')),
            color=alt.Color(
              'value:Q',
               scale=alt.Scale(
                    scheme='blueorange',
                    domain=[-1,1],
                    domainMid=0,
                    ),
               legend=alt.Legend(title=['Feature','Importance'], titleFontSize=fontSize, labelFontSize=fontSize)

            ),
            tooltip=[
                 alt.Tooltip(field='name',type='nominal', title='Feature'),
                 alt.Tooltip(field='value',type='quantitative', title='Importance')
                             ]
        )

        line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule().encode(x='x')

        text = alt.Chart(dataToPlot).transform_calculate(
            minSelection="toNumber(sel.xValue[0])",
            maxSelection="sel.x[1]"
        ).mark_text(
            align='left',
            baseline='top',
        ).encode(
            x=alt.value(5),
            y=alt.value(5),
            text=alt.Text("minSelection:N"),
        )

        step = fontSize*1.5
        #layering feature importnace chart
        featuresChart=(bars+line)

        #composing the chart
        finalChart=(dotSelection & featuresChart).properties(
            padding=10,
            ).configure_axis(
            labelLimit=step*15,
            labelFontSize=fontSize
        )

        return finalChart

    def plot_shap_values_alt2(self, feature_names, exp, fontDimension=10):
        #data prepraration
        arr=np.c_[np.array(feature_names),exp]
        dataToPlot=pd.DataFrame(arr,columns=['name','value'])

        fontSize = fontDimension

        #selector
        slider = alt.binding_range(min=0, max=1, step=0.005, name='Importance cutoff value (±) ')
        selector = alt.selection_single(name="Cutter", fields=['cutoff'], bind=slider, init={'cutoff': 0.05})

        #charting
        bar= alt.Chart(
            dataToPlot,
            title='Shap Feature importance'
        ).transform_filter(
            'datum.value > -1 & datum.value < 1'
        ).transform_filter(
             (alt.datum.value > selector.cutoff ) | (alt.datum.value < -(selector.cutoff))
        ).mark_bar().encode(
            x=alt.X('value:Q',title=None),
            y=alt.Y('name:N',title=None, sort=alt.EncodingSortField(field='value', op='mean',order='descending')),
            color=alt.Color(
              'value:Q',
               scale=alt.Scale(
                    scheme='blueorange',
                    domain=[-1,1],
                    domainMid=0,
                    ),
              legend=alt.Legend(title=['Feature','Importance'], titleFontSize=fontSize, labelFontSize=fontSize)
                ),
            tooltip=[
                 alt.Tooltip(field='name',type='nominal', title='Feature'),
                 alt.Tooltip(field='value',type='quantitative', title='Importance')
                             ]
        ).add_selection(
        selector
        )
        line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule().encode(x='x')


        step = fontSize*1.5

        chart=(bar+line).properties(
            width=450,
            height=alt.Step(step=step),
            padding=10,
            ).configure_axis(
            labelLimit=step*15,
            labelFontSize=fontSize
        )

        # from IPython.display import HTML
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
        """ % (fontSize)))
        display(chart)
