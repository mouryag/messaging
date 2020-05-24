from django.db import models

# Create your models here.
class ChatBox(models.Model):
    message=models.CharField(max_length=500)
    def __str__(self):
        return self.message
class person(models.Model):
    name=models.CharField(max_length=30)
    def __str__(self):
        return self.name
