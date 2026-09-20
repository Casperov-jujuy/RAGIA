import io
import unittest
from fastapi import HTTPException, UploadFile

from app.ingestion.validator import validate_file_extension


class TestFileValidator(unittest.TestCase):
    """Suite de pruebas unitarias para el validador estricto de extensiones."""

    def _create_upload_file(self, filename: str) -> UploadFile:
        return UploadFile(filename=filename, file=io.BytesIO(b"dummy content"))

    def test_valid_extensions(self):
        """Verifica que .md y .pdf sean aceptados (incluyendo mayúsculas)."""
        test_cases = [
            ("documento.pdf", ".pdf"),
            ("MANUAL.PDF", ".pdf"),
            ("notas.md", ".md"),
            ("README.MD", ".md"),
            ("reporte.anual.2024.pdf", ".pdf"),
            ("seccion.resumen.md", ".md"),
        ]
        for filename, expected in test_cases:
            with self.subTest(filename=filename):
                file = self._create_upload_file(filename)
                result = validate_file_extension(file)
                self.assertEqual(result, expected)

    def test_invalid_extensions_rejected(self):
        """Verifica que cualquier formato que no sea .md o .pdf sea rechazado con HTTP 400."""
        invalid_files = [
            "foto.png",
            "imagen.jpg",
            "grafico.jpeg",
            "documento.docx",
            "hoja.xlsx",
            "archivo.txt",
            "presentacion.pptx",
            "script.py",
            "app.exe",
            "archivo_sin_extension",
            "",
            "   ",
        ]
        for filename in invalid_files:
            with self.subTest(filename=filename):
                file = self._create_upload_file(filename)
                with self.assertRaises(HTTPException) as ctx:
                    validate_file_extension(file)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertTrue(
                    "no permitido" in ctx.exception.detail or "no tiene un nombre válido" in ctx.exception.detail
                )


if __name__ == "__main__":
    unittest.main()
