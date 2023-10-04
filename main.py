import argparse
from AnimeFaces import logger
from AnimeFaces.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from AnimeFaces.pipeline.stage_02_prepare_model import PrepareModelPipeline
from AnimeFaces.pipeline.stage_03_train_model import TrainModelPipeline
from AnimeFaces.pipeline.stage_04_generate import GenerateFacePipeline
# from AnimeFaces.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

def main(config):
    STAGE_NAME = "Data Ingestion Stage"
    if config.data_ingestion:
        try:
            logger.info(f">>>>> stage {STAGE_NAME} STARTED <<<<<")
            data_ingestion_pipeline = DataIngestionPipeline()
            data_ingestion_pipeline.main()
            logger.info(f">>>>> stage {STAGE_NAME} COMPLETED <<<<<")
        except Exception as e:
            logger.exception(e)
    else:
        logger.info(f">>>>> stage {STAGE_NAME} SKIPPED <<<<<")
    
    STAGE_NAME = "Prepare Model Stage"
    if config.create_model:
        try:
            logger.info(f">>>>> stage {STAGE_NAME} STARTED <<<<<")
            prepare_model_pipeline = PrepareModelPipeline()
            prepare_model_pipeline.main(show=config.show_img)
            logger.info(f">>>>> stage {STAGE_NAME} COMPLETED <<<<<")
        except Exception as e:
            logger.exception(e)    
    else:
        logger.info(f">>>>> stage {STAGE_NAME} SKIPPED <<<<<")
    
    STAGE_NAME = "Model Training Stage"    
    if config.train:
        try:
            logger.info(f">>>>> stage {STAGE_NAME} STARTED <<<<<")
            train_model_pipeline = TrainModelPipeline(config.epochs, config.lr)
            train_model_pipeline.main(show=config.show_img)
            logger.info(f">>>>> stage {STAGE_NAME} COMPLETED <<<<<")
        except Exception as e:
            logger.exception(e)
    else:
        logger.info(f">>>>> stage {STAGE_NAME} SKIPPED <<<<<")

    STAGE_NAME = "Generate Anime Face Stage"    
    try:
        logger.info(f">>>>> stage {STAGE_NAME} STARTED <<<<<")
        generate_face_pipeline = GenerateFacePipeline()
        generate_face_pipeline.main(show=config.show_img)
        logger.info(f">>>>> stage {STAGE_NAME} COMPLETED <<<<<")
    except Exception as e:
            logger.exception(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate High Quality Anime Character Faces with noise')
    parser.add_argument('--data_ingestion', type=bool, default=False,
                            help='Run Data Ingestion. Default: False')
    parser.add_argument('--create_model', type=bool, default=False,
                            help='Create and Train model from scratch. Default: False')
    parser.add_argument('--show_img', type=bool, default=False,
                            help='Always Show Output Image. Default: False')
    parser.add_argument('--train', type=bool, default=False,
                            help='Train the base model. Default: False')
    parser.add_argument('--epoch', type=int, default=10,
                            help='Number of Epochs. Default: Epochs: 10')
    parser.add_argument('--lr', type=float, default=0.002,
                            help='Learning Rate. Default: Learning Rate: 0.002')
            
    configuration = parser.parse_args()
    main(configuration)