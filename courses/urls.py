from django.urls import path

from . import views

app_name = "courses"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("courses/<int:pk>/", views.detail, name="detail"),
    path("courses/<int:pk>/files/<int:file_pk>/", views.file_detail, name="file_detail"),
    path("courses/<int:pk>/files/<int:file_pk>/delete/", views.file_delete, name="file_delete"),
]
