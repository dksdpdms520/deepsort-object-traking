from django.urls import path

from . import views

from django.conf.urls.static import static
from django.conf import  settings

app_name = 'deepsort'

urlpatterns = [
    path('', views.test, name='test'),
    path('upload/', views.upload, name='upload'),
    path('check/', views.check, name='check'),
    path('view/', views.view_images, name='view_images'),
    path('modify/', views.modify_images, name='modify_images'),
    path('run/<slug:id>/', views.run, name='run'),
    path('save/', views.save_to_database, name='save')
]
