
class ModeConfig:
    MODE_0_WEIGHT = 1.0      # Weight for standard translation (no additional entity focus)
    MODE_1_WEIGHT = 2.0      # Weight factor for entity-aware translation (extra focus on entities)
    MODE_2_WEIGHT = 0.5      # Weight factor for placeholder-based named entity handling --- FOR THE FUTURE
    MODE_3_NER_WEIGHT = 0.5  # Weight for auxiliary NER loss in multi-task learning mode --- FOR THE FUTURE 
