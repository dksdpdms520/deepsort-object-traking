from pathlib import Path
from asgiref.sync import async_to_sync
from django.shortcuts import render, redirect



BASE_DIR = str(Path(__file__).resolve().parent.parent)



def test(request):
    return render(request, 'deepsort/test.html')


def upload(request):
    return render(request, 'deepsort/upload.html', {'form': 1})


def modify_images(request):

    return redirect('deepsort:test')


def view_images(request):
    context = {'videos': 1}

    return render(request, 'deepsort/select_view.html', context)


def check(request):
    context ={
        'videos':1,
    }

    return render(request, 'deepsort/check.html', context)


async def run(request, id):
    return redirect('deepsort:test')


def save_to_database(request):
    context = {'file_path': 1}

    return render(request, 'deepsort/save.html', context)