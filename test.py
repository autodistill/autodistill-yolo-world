from autodistill_yolo_world import YOLOWorld
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = YOLOWorld(ontology=CaptionOntology({"dog": "dog"}))

# label all images in a folder called `context_images`
result = base_model.predict("dog.png")

print(result)