from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

num_trees = 20
max_depth = 20
random_state = 0


class RFModel:

    def __init__(self):

        self.tree_n_estimators = num_trees
        self.tree_max_depth = max_depth

    def get_pipeline(self):
        rf = RandomForestClassifier(
            n_estimators=self.tree_n_estimators,
            max_depth=self.tree_max_depth,
            random_state=random_state
        )
        steps = [("rf", rf)]
        return Pipeline(steps=steps)
