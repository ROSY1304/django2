# api/models.py
from django.db import models

class DatasetUpload(models.Model):
    # Usamos FileField para guardar el archivo
    file = models.FileField(upload_to='user_datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        # Esto es para que se vea bonito en el panel de admin
        return f"Dataset subido el {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"