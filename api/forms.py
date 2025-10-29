# api/forms.py
from django import forms
from .models import DatasetUpload

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = DatasetUpload
        fields = ('file',)
        widgets = {
            # Añadimos clases de Bootstrap para que se vea bonito
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv' # Aceptar solo archivos CSV
            })
        }
        labels = {
            'file': 'Sube tu archivo .csv'
        }