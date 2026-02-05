import logging
from typing import List

from src.utils.github_parser.github_parser_config import (
    ExtractedExample,
    load_config_file,
)
from src.utils.github_parser.github_repo_loader import TorchGitHubLoader
from src.utils.github_parser.cst_parser import CSTCodeParser


logger = logging.getLogger(__file__)


class DatasetBuilder:
    def __init__(
        self, loader: TorchGitHubLoader, parser: CSTCodeParser
    ) -> None:
        self.loader = loader
        self.parser = parser

    def run(
        self, target_repo: str, target_folder: str
    ) -> List[ExtractedExample]:
        all_data: List[ExtractedExample] = []

        files_from_repo = self.loader.repo_walk(
            target_repo,
            target_folder,
        )

        for file in files_from_repo.files_iterator:
            logger.debug("Обрабатываем файл %s", file.path)
            fetched_examples = self.parser.parse_python_module(
                file.content, file.path
            )
            # вот сюда прописать еще парсинг маркдауна
            if fetched_examples:
                logger.info(
                    "Получили %s examples in %s",
                    len(fetched_examples),
                    file.path,
                )
                all_data.extend(fetched_examples)

        return all_data


# прикрутить сюда аргс?

if __name__ == "__main__":
    config = load_config_file()
    loader = TorchGitHubLoader(config)
    parser = CSTCodeParser()
    pipeline = DatasetBuilder(loader, parser)
    try:
        dataset = pipeline.run(
            config.repo_for_analyzing, config.specific_folder_for_analyzing
        )

        print(f"Найдено примеров: {len(dataset)}")
        if dataset:
            print("-" * 50)
            print(f"Файл: {dataset[0].source_object_path}")
            print(f"Объект: {dataset[0].source_object_name}")
            print(f"Описание задачи: {dataset[0].task_description[:100]}...")
            print(f"Код:\n{dataset[0].solution_code[:150]}")
    except Exception as e:
        logger.critical(
            "Ошибка при выполнении пайплайна! %s", e, exc_info=True
        )
