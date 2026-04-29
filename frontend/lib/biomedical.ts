/**
 * Biomedical type vocabulary — shared between Dashboard breakdowns,
 * Graph legend, and Curation. Definitions are kept short (≤ 1 line)
 * for use as hover tooltips in cramped UI.
 */

export const ENTITY_TYPE_DESCRIPTIONS: Record<string, string> = {
  DISEASE:  'Conditions, disorders, syndromes',
  GENE:     'Coding sequences and genomic loci',
  PROTEIN:  'Translated products, enzymes, receptors',
  DRUG:     'Therapeutics, compounds with known activity',
  PATHWAY:  'Biological pathways and signalling cascades',
  COMPOUND: 'General chemical compounds',
  CONCEPT:  'Other biomedical concepts',
  ORGANISM: 'Species, strains, model organisms',
  PERSON:   'People (authors, clinicians, researchers)',
};

export const RELATIONSHIP_TYPE_DESCRIPTIONS: Record<string, string> = {
  TREATS:           'Drug or therapy used to treat the target',
  CAUSES:           'Source causes or induces the target',
  INHIBITS:         'Source inhibits or blocks the target',
  ACTIVATES:        'Source activates or upregulates the target',
  INTERACTS_WITH:   'Direct interaction (e.g. protein–protein binding)',
  ASSOCIATED_WITH:  'Statistical or observational association',
  PART_OF:          'Source is a subcomponent or member of the target',
  REGULATES:        'Source regulates expression or activity of the target',
  BINDS_TO:         'Source binds to a molecular target',
  ENCODES:          'Gene encodes the protein/RNA target',
  EXPRESSED_IN:     'Source is expressed in the target tissue or cell',
  LOCATED_IN:       'Cellular or anatomical localisation',
  TARGETED_BY:      'Source is targeted by a drug, antibody, or therapy',
  RELATED_TO:       'Generic biomedical relationship',
  INFLUENCES:       'One-way influence without specifying mechanism',
  CONTRIBUTES_TO:   'Source contributes to the development or state of target',
  PRECEDES:         'Temporal precedence (X happens before Y)',
  COEXPRESSED_WITH: 'Co-expressed with the target under similar conditions',
};

export function entityTypeTooltip(type: string): string | undefined {
  return ENTITY_TYPE_DESCRIPTIONS[type.toUpperCase()];
}

export function relationshipTypeTooltip(type: string): string | undefined {
  return RELATIONSHIP_TYPE_DESCRIPTIONS[type.toUpperCase()];
}
