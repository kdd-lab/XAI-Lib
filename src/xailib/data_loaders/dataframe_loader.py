from lore_explainer.datamanager import prepare_dataset


def prepare_dataframe(df, class_field):
    return prepare_dataset(df, class_field)
