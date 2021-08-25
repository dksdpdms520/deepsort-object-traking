from django.contrib import admin

from .models import Video


class VideosAdmin(admin.ModelAdmin):
    list_display = ('id', 'video')
    list_display_links = ('id', 'video')
    list_per_page = 25


admin.site.register(Video, VideosAdmin)
