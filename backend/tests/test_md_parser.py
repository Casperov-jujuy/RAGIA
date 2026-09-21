import unittest
from app.ingestion.md_parser import MarkdownParser


class TestMarkdownParser(unittest.TestCase):
    """Suite de pruebas unitarias para MarkdownParser."""

    def test_parse_structured_markdown(self):
        """Verifica la división correcta de un documento con múltiples encabezados."""
        sample_md = """# Título Principal
Este es el contenido de la introducción principal.

## Sección 1: Instalación
Para instalar el proyecto ejecuta pip install.

### Subsección 1.1: Requisitos
Se requiere Python 3.11+.

## Sección 2: Configuración
Configura las variables de entorno en el archivo .env.
"""
        sections = MarkdownParser.parse_text(sample_md, filename="guia.md")

        self.assertEqual(len(sections), 4)

        # Sección 1
        self.assertEqual(sections[0].title, "Título Principal")
        self.assertEqual(sections[0].level, 1)
        self.assertIn("contenido de la introducción principal", sections[0].content)
        self.assertEqual(sections[0].metadata["source"], "guia.md")

        # Sección 2
        self.assertEqual(sections[1].title, "Sección 1: Instalación")
        self.assertEqual(sections[1].level, 2)
        self.assertIn("pip install", sections[1].content)

        # Sección 3
        self.assertEqual(sections[2].title, "Subsección 1.1: Requisitos")
        self.assertEqual(sections[2].level, 3)
        self.assertIn("Python 3.11+", sections[2].content)

        # Sección 4
        self.assertEqual(sections[3].title, "Sección 2: Configuración")
        self.assertEqual(sections[3].level, 2)
        self.assertIn("variables de entorno", sections[3].content)

    def test_ignore_headers_inside_code_blocks(self):
        """Verifica que las líneas con '#' dentro de bloques de código no se interpreten como encabezados."""
        sample_md = """# Guía de Código

Aquí mostramos un script en Python:

```python
# Este es un comentario dentro de un bloque de código
def saludar():
    # Otro comentario
    print("Hola mundo")
```

Fin del bloque de código.

## Siguiente Sección
Texto de la siguiente sección.
"""
        sections = MarkdownParser.parse_text(sample_md, filename="codigo.md")

        # Debe haber exactamente 2 secciones: "Guía de Código" y "Siguiente Sección"
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].title, "Guía de Código")
        self.assertIn("# Este es un comentario", sections[0].content)
        self.assertIn("Fin del bloque de código", sections[0].content)

        self.assertEqual(sections[1].title, "Siguiente Sección")
        self.assertIn("Texto de la siguiente sección", sections[1].content)

    def test_text_before_first_header(self):
        """Verifica que el texto previo al primer título se capture como Introducción (nivel 0)."""
        sample_md = """Texto previo al primer encabezado.
Notas preliminares del autor.

# Primer Capítulo
Contenido del primer capítulo.
"""
        sections = MarkdownParser.parse_text(sample_md, filename="intro.md")

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].title, "Introducción")
        self.assertEqual(sections[0].level, 0)
        self.assertIn("Texto previo al primer encabezado", sections[0].content)

        self.assertEqual(sections[1].title, "Primer Capítulo")
        self.assertEqual(sections[1].level, 1)

    def test_markdown_without_headers(self):
        """Verifica que un archivo Markdown plano sin títulos se capture como una sola sección."""
        sample_md = "Este es un documento plano sin ningún encabezado.\nSolo contiene párrafos simples."
        sections = MarkdownParser.parse_text(sample_md, filename="plano.md")

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].title, "Documento")
        self.assertEqual(sections[0].level, 0)
        self.assertEqual(sections[0].content, sample_md)

    def test_empty_document(self):
        """Verifica que un documento vacío retorne una lista vacía."""
        self.assertEqual(MarkdownParser.parse_text(""), [])
        self.assertEqual(MarkdownParser.parse_text("    \n\n  "), [])


if __name__ == "__main__":
    unittest.main()
