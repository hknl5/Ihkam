from django.urls import path

from . import views

app_name = "courses"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("courses/<int:pk>/", views.detail, name="detail"),
    path("courses/<int:pk>/files/<int:file_pk>/", views.file_detail, name="file_detail"),
    path("courses/<int:pk>/files/<int:file_pk>/delete/", views.file_delete, name="file_delete"),
]

# M2 — topic review. Every edit is its own POST endpoint so each one is
# separately testable and none can be triggered by a link.
urlpatterns += [
    path("courses/<int:pk>/topics/", views.topics, name="topics"),
    path("courses/<int:pk>/topics/extract/", views.topics_extract, name="topics_extract"),
    path("courses/<int:pk>/topics/add/", views.topic_add, name="topic_add"),
    path("courses/<int:pk>/topics/merge/", views.topics_merge, name="topics_merge"),
    path("courses/<int:pk>/topics/<int:topic_pk>/rename/", views.topic_rename, name="topic_rename"),
    path("courses/<int:pk>/topics/<int:topic_pk>/delete/", views.topic_delete, name="topic_delete"),
    path("courses/<int:pk>/topics/<int:topic_pk>/exclude/", views.topic_exclude, name="topic_exclude"),
]
