/// Размер "виртуального" кэша: 4 ГиБ (2^30).
const LEN: usize = 1 << 32;
/// Размер «страницы» из точки зрения нашего кэша — 4 К (дробить меньше смысла).
const PAGE: usize = 64 * 1024;

// fn main() -> anyhow::Result<()> {
//     // 1) Резервируем 1 ГиБ VA. Физические страницы пока НЕ выделены.
//     let mut map: MmapMut = MmapOptions::new().len(LEN).map_anon()?;

//     println!("mmap created: {} MiB virtual ⇒ 0 MiB RAM (пока)", LEN / (1 << 20));

//     // 2) Пишем в ~16 случайных страниц; ОС «материализует» их лениво.
//     // let mut rng = SmallRng::seed_from_u64(0xCAFE_BABE);
//     let before = Instant::now();

//     for i in 0..16 {
//         let off = i * PAGE + 1;
//         map[off] = 42; // первая запись ⇒ page-fault ⇒ ядро отдаёт 64 К
//         println!("Touched page {:5} ⇒ commit +4 / 64 K", off / PAGE);
//         // sleep(Duration::from_secs(10));
//     }

//     println!("Done in {:.2?}", before.elapsed());
//     // sleep(Duration::from_secs(120));

//     Ok(())
// }