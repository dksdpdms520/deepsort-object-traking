import os
import random
import shutil

from django.shortcuts import redirect, render
from typing import NamedTuple


class Image(NamedTuple):
    file: str
    folder: str


class ImagesNotEnough(Exception):
    pass


def get_data(path):
    return os.listdir(path)


def home(request):
    return render(request, 'pages/home.html')


def move(request):
    PATH = 'static/pre_processing'
    image_files = []
    folders = get_data(PATH)

    for video_names in folders:
        video_name = f"{PATH}/{video_names}"
        person_tags = get_data(video_name)
        for tag in person_tags:
            image_tag = f"{video_name}/{tag}"
            images = get_data(image_tag)
            n = len(images)
            if n > 1:
                idx  = random.randint(0, n-1)
                # for image in images[-1:]:
                image_file = f"{image_tag}/{images[idx]}"
                image_files.append(Image(file=image_file[7:], folder=image_tag))
    random.shuffle(image_files)
    context = {'images': image_files}

    return render(request, 'pages/move.html', context)


def merge(request):
    if request.method == 'POST':
        images = request.POST.getlist('images')
        if len(images) >= 2:
            _to = images[0]
            targets = images[1:]
            for target in targets:
                images_files = os.listdir(target)
                for image in images_files:
                    try:
                        shutil.move(f"{target}/{image}", _to)
                    except shutil.Error:
                        pass
                shutil.rmtree(target)

        else:
            raise ImagesNotEnough("Images should be selected at least 2 images.")


    return redirect('pages:move')
