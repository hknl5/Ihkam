"""M0 success check, as code: auth works end to end. Needs the database.

The post-login landing page moved to the `courses` app in M1; see
`tests/test_courses.py` for what it now shows.
"""

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse


class AuthFlowTests(TestCase):
    def test_dashboard_requires_login(self):
        response = self.client.get(reverse("courses:dashboard"))
        self.assertRedirects(
            response, f"{reverse('accounts:login')}?next={reverse('courses:dashboard')}"
        )

    def test_signup_logs_in_and_lands_on_dashboard(self):
        response = self.client.post(
            reverse("accounts:signup"),
            {
                "username": "nadia",
                "first_name": "Nadia Al-Harbi",
                "email": "nadia@example.edu",
                "password1": "quiet-precision-42",
                "password2": "quiet-precision-42",
            },
            follow=True,
        )
        self.assertRedirects(response, reverse("courses:dashboard"))
        self.assertContains(response, "My courses")
        self.assertTrue(User.objects.filter(username="nadia").exists())

    def test_login_then_logout(self):
        User.objects.create_user("nadia", password="quiet-precision-42")

        login = self.client.post(
            reverse("accounts:login"),
            {"username": "nadia", "password": "quiet-precision-42"},
        )
        self.assertRedirects(login, reverse("courses:dashboard"))

        logout = self.client.post(reverse("accounts:logout"))
        self.assertRedirects(logout, reverse("accounts:login"))
        self.assertRedirects(
            self.client.get(reverse("courses:dashboard")),
            f"{reverse('accounts:login')}?next={reverse('courses:dashboard')}",
        )
