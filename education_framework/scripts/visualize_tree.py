# scripts/visualize_tree.py
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # if scripts/ is directly under education_framework/
MODEL_PATH = PROJECT_ROOT / "models" / "quality_trees_assistments.joblib"


def main(topic_id: int = 0):
    from models.quality_tree_bank import LeafQualityModel, QualityTreeBank
    import __main__
    __main__.LeafQualityModel = LeafQualityModel
    __main__.QualityTreeBank = QualityTreeBank
    bank = joblib.load(MODEL_PATH)
    model = bank["bank"][topic_id]  # LeafQualityModel
    feature_names = bank["feature_names"]

    plt.figure(figsize=(24, 12))
    plot_tree(
        model.tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        max_depth=None,        # set e.g. 4 if the plot is too large
        fontsize=10
    )
    plt.title(f"Decision Tree (topic {topic_id})")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / f"models/tree_topic_{topic_id}.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main(topic_id=0)
