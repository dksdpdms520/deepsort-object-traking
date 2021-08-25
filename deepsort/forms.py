from django import forms

from .models import Videos

class VideoForm(forms.ModelForm):
    class Meta:
        model = Videos
        fields=("title", "video")