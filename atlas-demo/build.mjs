import * as esbuild from 'esbuild';
import {mkdir,copyFile,writeFile} from 'node:fs/promises';
await mkdir('dist',{recursive:true});
await esbuild.build({entryPoints:['bootstrap.js'],bundle:true,format:'iife',platform:'browser',target:['es2020'],outfile:'dist/bundle.js',minify:true,logLevel:'info',loader:{'.png':'dataurl','.svg':'dataurl'}});
for(const name of ['index.html','style.css','asisa-theme.css'])await copyFile(name,'dist/'+name);
console.log('ATLAS standalone map engine bundled with app.');
