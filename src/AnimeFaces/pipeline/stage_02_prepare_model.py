from AnimeFaces.config.configuration import ConfigurationManager
from AnimeFaces.components.prepare_model import PrepareModel 
from AnimeFaces import logger

STAGE_NAME = "Prepare Model Stage"

class PrepareModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_model_config = config.get_prepare_model_config()
        prepare_model = PrepareModel(config=prepare_model_config)
        prepare_model.PrepareDiscriminatorModel()
        prepare_model.PrepareGeneratorModel()

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        prepare_model_training_pipeline = PrepareModelTrainingPipeline()
        prepare_model_training_pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)