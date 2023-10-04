from AnimeFaces.config.configuration import ConfigurationManager
from AnimeFaces.components.prepare_model import PrepareModel 
from AnimeFaces import logger

STAGE_NAME = "Train Model Stage"

class TrainModelPipeline:
    def __init__(self, epochs=10, lr=0.002):
        self.epochs = 10
        self.lr = 0.002

    def main(self, show):
        config = ConfigurationManager()
        train_model_config = config.get_train_model_config()
        train_model = TrainModel(config = train_model_config)
        train_model.load_models()
        train_model.load_and_transform_data()
        # history = train_model.fit(self.epochs, self.lr, show)
        train_model.save_trained_model(history)

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        train_model_pipeline = TrainModelPipeline()
        train_model_pipeline.main(show=False)
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)