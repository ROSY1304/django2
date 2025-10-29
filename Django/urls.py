from django.contrib import admin
from django.urls import path, include
from api import views as api_views # Importa la vista principal de tu app 'api'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')), # Incluye las URLs de tu app 'api'
    path('', api_views.index, name='home'), # Ruta para tu página de inicio con el formulario
]