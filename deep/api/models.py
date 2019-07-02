from django.db import models

class deepModel (models.Model):
    name = models.CharField(max_length=200)
    creation_date = models.DateTimeField(blank=True, null=True)
    num_classes = models.IntegerField()
    