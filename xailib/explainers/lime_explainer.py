import pandas as pd
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np

from xailib.models.bbox import AbstractBBox
from xailib.xailib_tabular import TabularExplainer
from xailib.xailib_image import ImageExplainer
from externals.lore.datamanager import prepare_dataset
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_image import LimeImageExplainer

class LimeXAITabularExplainer(TabularExplainer):
    lime_explainer = None

    def __init__(self, bb: AbstractBBox):
        super().__init__()
        self.bb = bb

    def fit(self, _df: pd.DataFrame, class_name, config):
        df, feature_names, class_values, numeric_columns, \
        _, _, _ = prepare_dataset(_df, class_name)
        feature_selection=config['feature_selection'] if 'feature_selection' in config else None
        discretize_continuous = config['discretize_continuous'] if 'discretize_continuous' in config else False
        discretizer = config['discretizer'] if 'discretizer' in config else 'quartile'
        sample_around_instance = config['sample_around_instance'] if 'sample_around_instance' in config else False
        kernel_width = config['kernel_width'] if 'kernel_width' in config else None
        kernel = config['kernel'] if 'kernel' in config else None
        self.lime_explainer = LimeTabularExplainer(df[feature_names].values, feature_names=feature_names,
                                                   class_names=class_values, feature_selection=feature_selection,
                                                   discretize_continuous=discretize_continuous, discretizer=discretizer,
                                                   sample_around_instance=sample_around_instance, kernel_width=kernel_width,
                                                   kernel=kernel)



    def explain(self, x, classifier_fn=None, num_samples=1000, top_labels=5):
        if classifier_fn:
            self.classifier_fn = classifier_fn
        else:
            self.classifier_fn = self.bb.predict_proba
        exp = self.lime_explainer.explain_instance(x, 
                                                   self.classifier_fn, 
                                                   num_samples=num_samples, 
                                                   top_labels=top_labels)
        return exp

    def plot_lime_values(self, exp, range_start, range_end):
        feature_names = [a_tuple[0] for a_tuple in exp]
        exp = [a_tuple[1] for a_tuple in exp]
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



class LimeXAIImageExplainer(ImageExplainer):
    lime_explainer = None

    def __init__(self, bb: AbstractBBox):
        """
        Arguments:
            bb: black box model
        """
        super().__init__()
        self.bb = bb

    def fit(self, verbose=False):
        """
        Create the explainer, extra parameters will be set in the explain function
        """
        self.lime_explainer = LimeImageExplainer(verbose = False)

    def explain(self, image, classifier_fn=None, top_labels=5, num_samples=1000):
        """
        Return LIME explanation
        Arguments: 
            image: query image to explain
            classifier_fn: [None] function that takes as input an array of images (the LIME neighbourhood) and return an array of (num_images,num_classes)
                           If None will use black_box.predict function
            top_labels: For multiclass problems select the best top_labels from the results to produce the explanation
            num_samples: number of points in the generated neighbourhood
        """
        if classifier_fn:
            self.classifier_fn = classifier_fn
        else:
            self.classifier_fn = self.bb.predict
            
        exp = self.lime_explainer.explain_instance(image,
                                                   self.classifier_fn,
                                                   top_labels=top_labels,
                                                   hide_color=0,
                                                   num_samples=num_samples)
        return exp
    
    def plot_lime_values(self, image, explanation, figsize=(15,5)):
        """
        Plot a three figure plot: [query image, heatmap of superpixels, overlap of the twos]
        Arguments:
            image: image to explain used in the explain function
            explanation: explanation returned by the explain function
            figsize: tuple of figure dimension
        """
        
        F, ax = plt.subplots(1,3,figsize=figsize)
        ax[0].imshow(image)
        ax[0].axis('off')
        ax[0].set_title('Query Image')
        
        #plot heatmap
        ind =  explanation.top_labels[0]
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
        ax[1].imshow(heatmap, cmap = 'coolwarm', vmin  = -heatmap.max(), vmax = heatmap.max())
        ax[1].axis('off')
        ax[1].set_title('Super Pixel Heatmap Explanation')
        
        #plot overlap
        ax[2].imshow(image)
        ax[2].imshow(heatmap,alpha=0.5,cmap='coolwarm')
        ax[2].axis('off')
        ax[2].set_title('Overlap of Query Image and Heatmap');
        
        
        
        
        