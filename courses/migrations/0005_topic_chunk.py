"""M2: the editable topic list, and one embedded passage per chunk of page text.

`Chunk.embedding` is a fixed-width pgvector column, baked from
`settings.EMBEDDING_DIM` at the moment this migration was written (1536).
Changing that setting later needs a new migration and a re-embed — the column
cannot silently follow it.
"""

import django.db.models.deletion
import pgvector.django.vector
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('courses', '0004_extractedpage_ocr_reason'),
        # The vector column needs `CREATE EXTENSION vector` to have run.
        ('agents', '0001_enable_pgvector'),
    ]

    operations = [
        migrations.CreateModel(
            name='Topic',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=300)),
                ('page_start', models.PositiveIntegerField(blank=True, null=True)),
                ('page_end', models.PositiveIntegerField(blank=True, null=True)),
                ('excluded', models.BooleanField(default=False, help_text='Not taught in lectures — never used to generate questions.')),
                ('key_terms', models.JSONField(blank=True, default=list)),
                ('definitions', models.JSONField(blank=True, default=list)),
                ('formulas', models.JSONField(blank=True, default=list)),
                ('examples', models.JSONField(blank=True, default=list)),
                ('position', models.PositiveIntegerField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('course', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='topics', to='courses.course')),
                ('parent', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='subtopics', to='courses.topic')),
                ('source_file', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='topics', to='courses.sourcefile')),
            ],
            options={
                'ordering': ['position', 'pk'],
            },
        ),
        migrations.CreateModel(
            name='Chunk',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('page', models.PositiveIntegerField()),
                ('position', models.PositiveIntegerField(default=0)),
                ('text', models.TextField()),
                ('embedding', pgvector.django.vector.VectorField(dimensions=1536)),
                ('source', models.CharField(choices=[('text_layer', 'Text layer'), ('ocr', 'Read by OCR')], default='text_layer', max_length=12)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('source_file', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='chunks', to='courses.sourcefile')),
                ('topic', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='chunks', to='courses.topic')),
            ],
            options={
                'ordering': ['source_file_id', 'page', 'position'],
                'constraints': [models.UniqueConstraint(fields=('source_file', 'page', 'position'), name='unique_chunk_position')],
            },
        ),
    ]
