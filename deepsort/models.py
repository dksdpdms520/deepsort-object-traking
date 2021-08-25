from django.urls import reverse
from django.db import models
 

def image_path(instance, filename):
    return 'images/{0}/{1}/{1}_{2}'.format(instance.video.title, instance.person, filename)
 

class Video(models.Model):
    title = models.CharField(max_length=100)
    video = models.FileField(upload_to='videos/%y')
     
    class Meta:
        verbose_name = 'video'
        verbose_name_plural = 'videos'
         
    def __str__(self):
        return str(self.title)

    def get_absolute_url(self):
        return reverse("deepsort:run", args=[self.id])


class Image(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    person = models.CharField(max_length=50)
    coordinates = models.CharField(max_length=10)
    file = models.FileField(upload_to=image_path)
    
    class Meta:
        verbose_name = 'image'
        verbose_name_plural = 'images'
