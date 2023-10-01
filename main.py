import argparse
from AnimeFaces import logger
from AnimeFaces.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
# from AnimeFaces.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
# from AnimeFaces.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
# from AnimeFaces.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
# from AnimeFaces.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

def main(config):
    STAGE_NAME = "Data Ingestion Stage"
    if config.data_ingestion:
        try:
            logger.info(f">>>>> stage {STAGE_NAME} STARTED <<<<<")
            data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
            data_ingestion_training_pipeline.main()
            logger.info(f">>>>> stage {STAGE_NAME} COMPLETED <<<<<")
        except Exception as e:
            logger.exception(e)
    else:
        logger.info(f">>>>> stage {STAGE_NAME} SKIPPED <<<<<")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate High Quality Anime Character Faces with noise')
    parser.add_argument('--data_ingestion', type=bool, default=False,
                            help='Run Data Ingestion. Default: False')
    configuration = parser.parse_args()
    main(configuration)