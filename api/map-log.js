export default function handler(req,res){
  if(req.method!=='POST'){res.setHeader('Allow','POST');return res.status(405).json({ok:false})}
  try{
    const b=typeof req.body==='string'?JSON.parse(req.body||'{}'):(req.body||{});
    const safe={
      session:String(b.session||'').slice(0,80),
      event:String(b.event||'').slice(0,120),
      detail:b.detail&&typeof b.detail==='object'?b.detail:{value:String(b.detail||'').slice(0,1000)},
      href:String(b.href||'').replace(/([?&]_vercel_share=)[^&]+/,'$1[redacted]').slice(0,500),
      ua:String(req.headers['user-agent']||'').slice(0,400),
      ts:String(b.ts||new Date().toISOString())
    };
    console.log('ATLAS_MAP_TRACE',JSON.stringify(safe));
    return res.status(200).json({ok:true});
  }catch(e){
    console.error('ATLAS_MAP_TRACE_PARSE',e?.message||String(e));
    return res.status(200).json({ok:false});
  }
}