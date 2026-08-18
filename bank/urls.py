from django.urls import path

from . import views

app_name = "bank"

# M12 — the bank is per course, so every url here starts from one. Search is a
# POST to the browse url rather than a url of its own: it is the same screen,
# ranked, and an embedding call is not something a link should spend.
urlpatterns = [
    path("courses/<int:pk>/bank/", views.bank_browse, name="browse"),
    path("courses/<int:pk>/bank/<int:bank_pk>/use/", views.bank_use, name="use"),
]
