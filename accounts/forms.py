from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class InstructorSignUpForm(UserCreationForm):
    """Django's built-in signup, with the fields an instructor record needs."""

    first_name = forms.CharField(max_length=150, required=True, label="Full name")
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "first_name", "email")

    def clean_email(self):
        email = self.cleaned_data["email"]
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account already uses this email address.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        user.first_name = self.cleaned_data["first_name"]
        if commit:
            user.save()
        return user
