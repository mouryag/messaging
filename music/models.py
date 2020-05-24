from django.db import models

# Create your models here.
class Album(models.Model):
    artist=models.CharField(max_length=250)
    album_title=models.CharField(max_length=500)
    genre=models.CharField(max_length=200)
    album_logo=models.CharField(max_length=1000)
    def __str__(self):
        return self.album_title+"-"+self.artist
class Song(models.Model):
    album=models.ForeignKey(Album,on_delete=models.CASCADE)
    file_type=models.CharField(max_length=20)
    song_title = models.CharField(max_length=100)
    is_favorite=models.BooleanField(default=False)
    def __str__(self):
        return self.song_title
# album1 is object of Album with pk=1 and album1.song_set.all() gives all songs in that album.
# album1.song_set.create(song_titile=....,...) can create new song in that album,,  album1.song_set.count()
