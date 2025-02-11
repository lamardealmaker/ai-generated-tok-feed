import pytest
from src.director.variation.template_engine import TemplateLibrary

def test_template_count():
    """Verify the number of unique templates."""
    library = TemplateLibrary()
    templates = library.list_templates()
    
    # Count unique combinations
    styles = set()
    pacings = set()
    structures = set()
    color_schemes = set()
    
    for template in templates:
        parts = template.split('_')
        styles.add(parts[0])
        pacings.add(parts[1])
        structures.add(parts[2])
        color_schemes.add(parts[3])
    
    print(f"\nTemplate Statistics:")
    print(f"Total Templates: {len(templates)}")
    print(f"Unique Styles: {len(styles)} - {sorted(styles)}")
    print(f"Unique Pacings: {len(pacings)} - {sorted(pacings)}")
    print(f"Unique Structures: {len(structures)} - {sorted(structures)}")
    print(f"Unique Color Schemes: {len(color_schemes)} - {sorted(color_schemes)}")
    
    # We expect:
    # 5 styles × 3 pacings × 5 structures × 3 color schemes = 225 templates
    assert len(templates) == 225
    assert len(styles) == 5
    assert len(pacings) == 3
    assert len(structures) == 5
    assert len(color_schemes) >= 3  # Each style has 3 color schemes
