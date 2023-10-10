import supervisely as sly
import os
import csv
from collections import defaultdict
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name, get_file_size
import shutil

from tqdm import tqdm


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(
                        team_id, teamfiles_path, local_path, progress_cb=pbar
                    )

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###

    batch_size = 30

    dataset_path = "carbv"
    test_folder = os.path.join("test", "test")
    train_folder = "archive (2)"
    images_folder = os.path.join(train_folder, "train_images")
    masks_folder = os.path.join(train_folder, "train_masks")
    metadata_csv = os.path.join("carbv", "metadata.csv")
    images_ext = ".jpg"
    masks_ext = ".png"
    ds_name = "train"
    group_tag_name = "car_id"

    ds_img_info = defaultdict()
    with open(metadata_csv) as file:
        csvreader = csv.reader(file)
        for idx, row in enumerate(csvreader):
            if idx == 0:
                ds_img_info_names = [name for name in row]
                continue
            ds_img_info[row[0]] = {
                name: row[i] for i, name in enumerate(ds_img_info_names)
            }

    tag_metas = [
        sly.TagMeta(name, value_type=sly.TagValueType.ANY_STRING)
        for name in ds_img_info_names
    ]

    def create_ann(image_path):
        labels = []
        tags = []
        img_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = img_np.shape[0]
        img_wight = img_np.shape[1]
        image_name = get_file_name(image_path)
        id_data = image_name.split("_")[0]
        tags_info = ds_img_info[id_data]
        for tag in tags_info:
            tag_sly = [
                sly.Tag(meta=tag_meta, value=ds_img_info[id_data][tag_meta.name])
                for tag_meta in tag_metas
                if tag_meta.name == tag
            ]
            tags.extend(tag_sly)
        # group_id = sly.Tag(tag_id, value=id_data)
        if ds_name != "test":
            mask_name = image_name + masks_ext
            mask_path = os.path.join(masks_path, mask_name)
            mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
            mask = mask_np == 1
            curr_bitmap = sly.Bitmap(mask)
            curr_label = sly.Label(curr_bitmap, obj_class)
            labels.append(curr_label)

        return sly.Annotation(
            img_size=(img_height, img_wight), labels=labels, img_tags=tags
        )

    obj_class = sly.ObjClass("car", sly.Bitmap)

    tag_metas = [
        sly.TagMeta(name, value_type=sly.TagValueType.ANY_STRING)
        for name in ds_img_info_names
    ]

    project = api.project.create(
        workspace_id, project_name, change_name_if_conflict=True
    )
    meta = sly.ProjectMeta(obj_classes=[obj_class], tag_metas=tag_metas)
    api.project.update_meta(project.id, meta.to_json())

    images_path = os.path.join(dataset_path, images_folder)
    masks_path = os.path.join(dataset_path, masks_folder)

    images_names = os.listdir(images_path)

    progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))
    train_path = os.path.join("carbv", "archive (2", "train_images")
    test_path = os.path.join("carbv", "test", "test")
    proj_dict = {"train": train_path, "test": test_path}

    for ds_name in proj_dict:
        dataset = api.dataset.create(
            project.id, ds_name.lower(), change_name_if_conflict=True
        )
        images_names = [name for name in os.listdir(proj_dict[ds_name])]
        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(proj_dict[ds_name], image_name)
                for image_name in img_names_batch
            ]

            img_infos = api.image.upload_paths(
                dataset.id, img_names_batch, images_pathes_batch
            )
            img_ids = [im_info.id for im_info in img_infos]
            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))

    return project
