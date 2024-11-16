
import pandas as pd

data = {
    "Mode": [0, 1, 2, 3],
    "Behavior": [
        "Standard Translation",
        "Entity-Aware Translation",
        "Placeholder-Based",
        "Auxiliary NER Loss"
    ],
    "Loss Function": [
        "Cross-Entropy Loss",
        "entity_aware_loss (weighted for entities)",
        "placeholder_loss",
        "ner_auxiliary_loss (multi-task learning)"
    ],
    "Focus": [
        "None",
        "Named entities emphasized",
        "Named entities treated as placeholders",
        "Enhanced NER attention"
    ]
}

table_df = pd.DataFrame(data)

import ace_tools as tools; tools.display_dataframe_to_user(name="Mode Comparison Table", dataframe=table_df)
