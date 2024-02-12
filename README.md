<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill YOLO World Module

This repository contains the code supporting the YOLO World base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[CLIP](https://github.com/openai/CLIP), developed by OpenAI, is a computer vision model trained using pairs of images and text. You can use CLIP with autodistill for image classification.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [YOLO World Autodistill documentation](https://autodistill.github.io/autodistill/base_models/yolo_world/).

## Installation

To use YOLO World with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-yolo-world
```

## Quickstart

```python
from autodistill_clip import CLIP

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = CLIP(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpeg")
```


## License

[add license information here]

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!