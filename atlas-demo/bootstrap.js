import maplibregl from 'maplibre-gl';
import {MapboxOverlay} from '@deck.gl/mapbox';
import {ColumnLayer,ScatterplotLayer,ArcLayer} from '@deck.gl/layers';
import {HeatmapLayer} from '@deck.gl/aggregation-layers';
import 'maplibre-gl/dist/maplibre-gl.css';
// Bundle with ATLAS: the map no longer needs third-party JS CDNs at runtime.
window.maplibregl=maplibregl;
window.deck={MapboxOverlay,ColumnLayer,ScatterplotLayer,ArcLayer,HeatmapLayer};
import './app.js';
