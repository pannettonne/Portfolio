import * as esbuild from 'esbuild';
import {mkdir,copyFile,writeFile,readFile} from 'node:fs/promises';
import {feature} from 'topojson-client';
await mkdir('dist',{recursive:true});
const topology=JSON.parse(await readFile(new URL('./node_modules/world-atlas/countries-10m.json',import.meta.url),'utf8'));
const countries=feature(topology,topology.objects.countries).features;
const peninsula=countries.filter(f=>['724','620'].includes(String(f.id))||['Spain','Portugal'].includes(f.properties?.name));
if(peninsula.length<2)throw new Error('The local map must contain Spain and Portugal');
await writeFile('dist/iberia.geojson',JSON.stringify({type:'FeatureCollection',features:peninsula}));
console.log('ATLAS: offline Iberian geographical contours built: '+peninsula.length+' countries');

const escapeXML=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
const x=lon=>(lon+10.5)*80+58, y=lat=>(44.5-lat)*69+30;
function ringToPath(ring){return ring.map((p,i)=>(i?'L':'M')+x(p[0]).toFixed(1)+' '+y(p[1]).toFixed(1)).join(' ')+'Z'}
function shapePath(geometry){return geometry.type==='Polygon'?geometry.coordinates.map(ringToPath).join(' '):geometry.coordinates.flatMap(p=>p.map(ringToPath)).join(' ')}
const places=[['Madrid',-3.70,40.42,225],['Barcelona',2.17,41.39,172],['Valencia',-.38,39.47,121],['Sevilla',-5.98,37.39,167],['Málaga',-4.42,36.72,138],['Bilbao',-2.93,43.26,119],['Zaragoza',-.88,41.65,106],['A Coruña',-8.41,43.36,86],['Valladolid',-4.73,41.65,78],['Murcia',-1.13,37.99,71]];
const major=places[0];
const routes=places.slice(1).map(p=>'<path d="M'+x(major[1])+','+y(major[2])+' Q'+((x(major[1])+x(p[1]))/2)+','+(Math.min(y(major[2]),y(p[2]))-75)+' '+x(p[1])+','+y(p[2])+'" fill="none" stroke="#3389c1" opacity=".45" stroke-dasharray="5 5" stroke-width="1.5"/>').join('');
const pins=places.map(p=>'<g><circle cx="'+x(p[1])+'" cy="'+y(p[2])+'" r="15" fill="#1c7aa6" opacity=".14"/><circle cx="'+x(p[1])+'" cy="'+y(p[2])+'" r="5.2" fill="#1784b6" stroke="white" stroke-width="2"/><text x="'+(x(p[1])+9)+'" y="'+(y(p[2])-8)+'" fill="#1a4b74" font-size="13" font-weight="700">'+escapeXML(p[0])+'</text></g>').join('');
const land=peninsula.map(f=>'<path d="'+shapePath(f.geometry)+'" fill="#e1f0ee" stroke="#77a7bd" stroke-width="1.7" fill-rule="evenodd"/>').join('');
const svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 660" role="img" aria-label="Mapa de España con red de centros ficticios"><defs><radialGradient id="sea"><stop stop-color="#f3faff"/><stop offset="1" stop-color="#d9eaf5"/></radialGradient></defs><rect width="1200" height="660" fill="url(#sea)"/><g>'+land+'</g><g>'+routes+pins+'</g><text x="53" y="624" font-size="13" fill="#6588a1">Mapa local de respaldo · Geometría Natural Earth · Actividad y centros ficticios</text></svg>';
await writeFile('dist/iberia-preview.svg',svg);
console.log('ATLAS: in-app geographical preview generated');

await esbuild.build({entryPoints:['bootstrap.js'],bundle:true,format:'iife',platform:'browser',target:['es2020'],outfile:'dist/bundle.js',minify:true,logLevel:'info',loader:{'.png':'dataurl','.svg':'dataurl'}});
for(const name of ['index.html','style.css','asisa-theme.css','app.js'])await copyFile(name,'dist/'+name);
console.log('ATLAS standalone map engine bundled with app.');
