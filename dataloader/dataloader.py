"""Dataset preprocessing and dataloader construction for AMSG."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import BertTokenizer

from utils.helper import collater, json_load


DATASET_JMERE = "JMERE"
DATASET_MNRE = "MNRE"
DATASET_TWITTER15 = "twitter15"
DATASET_TWITTER17 = "twitter17"

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def load_image_tensor(
    image_size: int,
    image_path: Union[os.PathLike, str],
) -> torch.Tensor:
    """Load and normalize one image in the format expected by the model."""

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD),
        ]
    )
    return transform(image).unsqueeze(0)


class MultimodalDataset(Dataset):
    """Dataset that tokenizes text, transforms labels, and loads images."""

    def __init__(
        self,
        data: Sequence[Tuple[Any, ...]],
        sim_mode: str,
        image_root: Union[os.PathLike, str],
        bert_local_path: str,
        image_size: int = 384,
        never_split_tokens: Optional[Sequence[str]] = None,
        dataset_name: str = DATASET_JMERE,
    ) -> None:
        self.data = data
        self.sim_mode = sim_mode.lower()
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_local_path,
            never_split=never_split_tokens,
        )

        if self.sim_mode not in {"itc", "itm"}:
            raise ValueError("sim_mode must be either 'itc' or 'itm'.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        words, itm_score, itc_score, ner_labels, relation_labels, image_name = self.data[index]

        sentence = " ".join(words)
        bert_length = len(self.tokenizer.tokenize(sentence)) + 2
        word_to_bert = self._map_words_to_bert_tokens(words)

        if self.dataset_name == DATASET_JMERE:
            ner_labels = self._transform_entity_labels(ner_labels, word_to_bert)
            relation_labels = self._transform_relation_labels(relation_labels, word_to_bert)
        elif self.dataset_name == DATASET_MNRE:
            ner_labels = None
            relation_labels = self._transform_relation_labels(relation_labels, word_to_bert)
        elif self.dataset_name in {DATASET_TWITTER15, DATASET_TWITTER17}:
            ner_labels = self._transform_entity_labels(ner_labels, word_to_bert)
            relation_labels = None
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        image = load_image_tensor(self.image_size, self.image_root / image_name)
        similarity_score = itc_score if self.sim_mode == "itc" else itm_score

        # The tuple order is consumed by utils.helper.collater.
        return (
            words,
            ner_labels,
            relation_labels,
            bert_length,
            similarity_score,
            image,
            sentence,
        )

    def _map_words_to_bert_tokens(self, words: Sequence[str]) -> Dict[int, List[int]]:
        word_to_bert: Dict[int, List[int]] = {}
        current_index = 0

        for word_index, word in enumerate(words):
            subwords = self.tokenizer.tokenize(word)
            word_to_bert[word_index] = [
                current_index,
                current_index + len(subwords) - 1,
            ]
            current_index += len(subwords)

        return word_to_bert

    @staticmethod
    def _transform_entity_labels(
        labels: Sequence[Any],
        word_to_bert: Dict[int, List[int]],
    ) -> List[Any]:
        transformed: List[Any] = []
        for index in range(0, len(labels), 3):
            start = word_to_bert[labels[index]][0] + 1
            end = word_to_bert[labels[index + 1]][0] + 1
            transformed.extend([start, end, labels[index + 2]])
        return transformed

    @staticmethod
    def _transform_relation_labels(
        labels: Sequence[Any],
        word_to_bert: Dict[int, List[int]],
    ) -> List[Any]:
        transformed: List[Any] = []
        for index in range(0, len(labels), 3):
            head = word_to_bert[labels[index]][0] + 1
            tail = word_to_bert[labels[index + 1]][0] + 1
            transformed.extend([head, tail, labels[index + 2]])
        return transformed


def preprocess_jmere(data: Sequence[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    """Convert JMERE JSON records to the internal dataset representation."""

    processed: List[Tuple[Any, ...]] = []
    for record in data:
        words = record["text"].split(" ")
        triples = record["triple_list"]
        entity_labels: List[Any] = []
        relation_labels: List[Any] = []

        if "ent_pair_list" in record:
            entity_pairs = record["ent_pair_list"]
            for triple_index, triple in enumerate(triples):
                subject = entity_pairs[triple_index][0][0]
                subject_end = entity_pairs[triple_index][0][1] - 1
                object_ = entity_pairs[triple_index][1][0]
                object_end = entity_pairs[triple_index][1][1] - 1

                if subject not in entity_labels:
                    entity_labels.extend([subject, subject_end, triple[3]])
                if object_ not in entity_labels:
                    entity_labels.extend([object_, object_end, triple[4]])
                relation_labels.extend([subject, object_, triple[1]])
        else:
            for triple in triples:
                subject = words.index(triple[0])
                object_ = words.index(triple[2])

                if subject not in entity_labels:
                    entity_labels.extend([subject, subject, "None"])
                if object_ not in entity_labels:
                    entity_labels.extend([object_, object_, "None"])
                relation_labels.extend([subject, object_, triple[1]])

        processed.append(
            (
                words,
                record["itm_score"],
                record["itc_score"],
                entity_labels,
                relation_labels,
                record["img_id"],
            )
        )

    return processed


def preprocess_twitter(data: Sequence[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    """Convert Twitter-2015/2017 JSON records to the internal representation."""

    processed: List[Tuple[Any, ...]] = []
    for record in data:
        words = record["text"].split(" ")
        entity_labels: List[Any] = []

        for entity in record["ents"]:
            start = entity["pos"][0]
            end = entity["pos"][1] - 1
            if start not in entity_labels:
                entity_labels.extend([start, end, entity["tag"]])

        processed.append(
            (
                words,
                record["itm_score"],
                record["itc_score"],
                entity_labels,
                None,
                record["img_id"],
            )
        )

    return processed


def preprocess_mnre(data: Sequence[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    """Convert MNRE JSON records to the internal representation."""

    processed: List[Tuple[Any, ...]] = []
    for record in data:
        words = record["text"].split(" ")
        triples = record["triple_list"]
        entity_pairs = record["ent_pair_list"]
        entity_labels: List[Any] = []
        relation_labels: List[Any] = []

        words.insert(entity_pairs[0][0][0], "<s>")
        words.insert(entity_pairs[0][0][1] + 1, "</s>")
        subject = entity_pairs[0][0][0]
        subject_end = entity_pairs[0][0][1] + 1

        object_start = entity_pairs[0][1][0]
        object_end = entity_pairs[0][1][1] + 1

        if entity_pairs[0][1][0] > entity_pairs[0][0][1]:
            object_start += 2
        elif entity_pairs[0][1][0] > entity_pairs[0][0][0]:
            object_start += 1
            subject_end += 1
        else:
            subject += 1
            subject_end += 1

        if entity_pairs[0][1][1] > entity_pairs[0][0][1]:
            object_end += 2
        elif entity_pairs[0][1][1] > entity_pairs[0][0][0]:
            subject_end += 1
            object_end += 1
        else:
            subject += 1
            subject_end += 1

        words.insert(object_start, "<o>")
        words.insert(object_end, "</o>")

        if subject not in entity_labels:
            entity_labels.extend([subject, subject_end, "None"])
        if object_start not in entity_labels:
            entity_labels.extend([object_start, object_end, "None"])
        relation_labels.extend([subject, object_start, triples[0][1]])

        processed.append(
            (
                words,
                record["itm_score"],
                record["itc_score"],
                entity_labels,
                relation_labels,
                record["img_id"],
            )
        )

    return processed


DATASET_CONFIGS = {
    DATASET_JMERE: {
        "data_dir": Path("datasets/JMERE/JMERE_new2"),
        "image_dir": Path("datasets/JMERE/JMERE_imgs"),
        "files": ("new_train_triples.json", "new_test_triples.json", "new_val_triples.json"),
        "preprocess": preprocess_jmere,
    },
    DATASET_MNRE: {
        "data_dir": Path("datasets/MNRE/mnre_txt_new2"),
        "image_dir": Path("datasets/MNRE/mnre_image"),
        "files": ("new_train_triples.json", "new_test_triples.json", "new_val_triples.json"),
        "preprocess": preprocess_mnre,
    },
    DATASET_TWITTER17: {
        "data_dir": Path("datasets/twitter17/twitter17_new2"),
        "image_dir": Path("datasets/twitter17/twitter2017_images"),
        "files": ("new_train.json", "new_test.json", "new_valid.json"),
        "preprocess": preprocess_twitter,
    },
    DATASET_TWITTER15: {
        "data_dir": Path("datasets/twitter15/twitter15_new2"),
        "image_dir": Path("datasets/twitter15/twitter2015_images"),
        "files": ("new_train.json", "new_test.json", "new_valid.json"),
        "preprocess": preprocess_twitter,
    },
}


def build_dataloaders(args, ner2idx, rel2idx):
    """Build train, development, and test dataloaders for one dataset."""

    if args.data not in DATASET_CONFIGS:
        supported = ", ".join(DATASET_CONFIGS)
        raise ValueError(f"Unsupported dataset '{args.data}'. Choose from: {supported}")

    config = DATASET_CONFIGS[args.data]
    data_dir = config["data_dir"]
    image_dir = config["image_dir"]
    train_file, test_file, dev_file = config["files"]
    preprocess = config["preprocess"]

    train_data = preprocess(json_load(str(data_dir), train_file))
    test_data = preprocess(json_load(str(data_dir), test_file))
    dev_data = preprocess(json_load(str(data_dir), dev_file))

    custom_tokens = ["<s>", "</s>", "<o>", "</o>"]
    is_twitter = args.data in {DATASET_TWITTER15, DATASET_TWITTER17}

    if is_twitter:
        train_image_dir = test_image_dir = dev_image_dir = image_dir
    else:
        train_image_dir = image_dir / "train"
        test_image_dir = image_dir / "test"
        dev_image_dir = image_dir / "val"

    train_dataset = MultimodalDataset(
        train_data,
        args.sim_mode,
        train_image_dir,
        args.bert_local_path,
        args.img_size,
        custom_tokens,
        args.data,
    )
    test_dataset = MultimodalDataset(
        test_data,
        args.sim_mode,
        test_image_dir,
        args.bert_local_path,
        args.img_size,
        custom_tokens,
        args.data,
    )
    dev_dataset = MultimodalDataset(
        dev_data,
        args.sim_mode,
        dev_image_dir,
        args.bert_local_path,
        args.img_size,
        custom_tokens,
        args.data,
    )

    collate_fn = collater(ner2idx, rel2idx, ifpos=False)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader, dev_loader
