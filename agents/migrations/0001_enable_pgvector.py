"""Enable the pgvector extension.

Lives in `agents` because the provider's `embed()` is what fills those columns;
the `chunks` table that uses them arrives in M2. Requires a database role with
permission to CREATE EXTENSION (superuser, or the extension pre-installed).
"""

from django.db import migrations
from pgvector.django import VectorExtension


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        VectorExtension(),
    ]
