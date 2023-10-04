from AnimeFaces.config.configuration import ConfigurationManager
from AnimeFaces.components.generate import Generate 
from AnimeFaces import logger

STAGE_NAME = "Generate Anime Faces Stage"

class GenerateFacePipeline:
    def __init__(self):
        pass

    def main(self, show):
        config = ConfigurationManager()
        generate_face_config = config.get_generate_config()
        generate_face = Generate(config = generate_face_config)
        generate_face.generate_image(show)

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        generate_face_pipeline = GenerateFacePipeline()
        generate_face_pipeline.main(show=True)
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)