LAND_COVER_ALIASES = {
    'needleleaf': [
        'needleleafforest', 'needleleafforests', 
        'needleleaf forest', 'needleleaf forests',
        'neadleleaf'],
    'broadleaf': [
        'broadleafforest', 'broadleafforests',
        'broadleaf forest', 'broadleaf forests',
        'broadleaf'],
    'shrubland': [
        'shrubland', 'shrublands', 'shrubs',
        'shrub', 'scrubland'],
    'grassland': [
        'grassland', 'grasslands', 'grass',
        'grasses', 'prairie', 'savanna'],
    'wetland': [
        'wetland', 'wetlands', 'marsh',
        'bog', 'swamp', 'fen'],
    'cropland': [
        'cropland', 'croplands', 'crop',
        'crops', 'agriculture', 'agricultural'],
    'barrenland': [
        'barrenland', 'barrenlands', 'barren',
        'barren', 'bare', 'bareland', 'barelands'],
    'urban': [
        'urban', 'city', 'cities',
        'built-up', 'suburban'],
    'water': [
        'water', 'waters', 'ocean', 'oceans',
        'sea', 'seas', 'lake', 'lakes', 'river',
        'rivers', 'stream', 'streams', 'slough',
        'sloughs', 'reservoir', 'reservoirs',
        'pond', 'ponds', 'canal', 'canals'],
    'snow': [
        'snow', 'ice', 'glacier', 'permafrost',
        'snowfield', 'snowfields', 'snowpack',
        'snow-and-ice', 'snow and ice', 'snow/ice',
        'snow/ice-covered', 'snow/ice covered', 'snow_and_ice',
        'snow-and_ice', 'snow and ice', 'snow-ice', 'snow ice',
        'snow_and-ice'],
}

_alias_lookup: dict[str, str] = {
    alias: canonical
    for canonical, aliases in LAND_COVER_ALIASES.items()
    for alias in aliases
}

def normalize_alias(word: str) -> str:
    return _alias_lookup.get(word.strip().lower(), word.strip().lower())