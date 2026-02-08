// INTERNECHIVE PDF BUILDER — Browser Console Script
// Works on any Internet Archive book page (archive.org/details/...)
//
// Usage:
//   1. Go to the book on archive.org and click "Borrow"
//   2. Make sure you can see a page in the reader
//   3. Open DevTools (F12) > Console
//   4. Paste this entire script and press Enter
//   5. Wait for it to finish — downloads a single .tar file

(async () => {
    var br = window.br;
    if (!br) { console.error('BookReader not found. Are you on an archive.org book page?'); return; }

    var bookId = br.bookId || 'book';
    var numLeafs = br.book.getNumLeafs();
    console.log('Book: ' + bookId + ' (' + numLeafs + ' pages)');
    console.log('Estimated time: ~' + Math.round(numLeafs * 4 / 60) + ' minutes. Do not close this tab.');

    var images = [];
    var failures = [];

    function findPageImage() {
        var allImgs = document.querySelectorAll('img');
        var best = null;
        var bestArea = 0;
        for (var j = 0; j < allImgs.length; j++) {
            var img = allImgs[j];
            if (img.src && img.src.indexOf('blob:') === 0 && img.naturalWidth > 200) {
                var area = img.naturalWidth * img.naturalHeight;
                if (area > bestArea) {
                    bestArea = area;
                    best = img;
                }
            }
        }
        return best;
    }

    function imgToBlob(img) {
        return new Promise(function(resolve) {
            var canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            canvas.toBlob(function(blob) {
                resolve(blob);
            }, 'image/jpeg', 0.95);
        });
    }

    for (var i = 0; i < numLeafs; i++) {
        var filename = 'page_' + String(i + 1).padStart(4, '0') + '.jpg';

        try {
            br.jumpToIndex(i);

            var img = null;
            for (var wait = 0; wait < 20; wait++) {
                await new Promise(function(r) { setTimeout(r, 500); });
                img = findPageImage();
                if (img && img.complete && img.naturalWidth > 200) break;
            }

            if (!img || !img.complete || img.naturalWidth < 200) {
                throw new Error('Page image not found or not loaded');
            }

            var blob = await imgToBlob(img);
            var arrayBuf = await blob.arrayBuffer();
            images.push({ name: filename, data: new Uint8Array(arrayBuf) });
            console.log('[' + (i + 1) + '/' + numLeafs + '] ' + filename + ' OK (' + Math.round(blob.size / 1024) + ' KB, ' + img.naturalWidth + 'x' + img.naturalHeight + ')');

        } catch (e) {
            console.error('[' + (i + 1) + '/' + numLeafs + '] ' + filename + ' FAILED: ' + e.message);
            failures.push(i + 1);
        }

        await new Promise(function(r) { setTimeout(r, 1500); });
    }

    console.log('Capture done. ' + images.length + ' succeeded, ' + failures.length + ' failed.');
    if (failures.length > 0) console.warn('Failed pages: ' + failures.join(', '));

    // Build tar file
    console.log('Building tar file...');

    function makeTarHeader(name, size) {
        var buf = new Uint8Array(512);
        for (var j = 0; j < name.length && j < 100; j++) buf[j] = name.charCodeAt(j);
        var mode = '0000644\0';
        for (var j = 0; j < 8; j++) buf[100 + j] = mode.charCodeAt(j);
        var uid = '0001000\0';
        for (var j = 0; j < 8; j++) buf[108 + j] = uid.charCodeAt(j);
        for (var j = 0; j < 8; j++) buf[116 + j] = uid.charCodeAt(j);
        var sizeStr = size.toString(8);
        while (sizeStr.length < 11) sizeStr = '0' + sizeStr;
        sizeStr += '\0';
        for (var j = 0; j < 12; j++) buf[124 + j] = sizeStr.charCodeAt(j);
        var mtime = Math.floor(Date.now() / 1000).toString(8);
        while (mtime.length < 11) mtime = '0' + mtime;
        mtime += '\0';
        for (var j = 0; j < 12; j++) buf[136 + j] = mtime.charCodeAt(j);
        for (var j = 148; j < 156; j++) buf[j] = 32;
        buf[156] = 48;
        var cksum = 0;
        for (var j = 0; j < 512; j++) cksum += buf[j];
        var cksumStr = cksum.toString(8);
        while (cksumStr.length < 6) cksumStr = '0' + cksumStr;
        cksumStr += '\0 ';
        for (var j = 0; j < 8; j++) buf[148 + j] = cksumStr.charCodeAt(j);
        return buf;
    }

    var totalSize = 0;
    for (var k = 0; k < images.length; k++) {
        totalSize += 512 + Math.ceil(images[k].data.length / 512) * 512;
    }
    totalSize += 1024;

    var tar = new Uint8Array(totalSize);
    var offset = 0;
    for (var k = 0; k < images.length; k++) {
        tar.set(makeTarHeader(images[k].name, images[k].data.length), offset);
        offset += 512;
        tar.set(images[k].data, offset);
        offset += Math.ceil(images[k].data.length / 512) * 512;
    }

    var tarBlob = new Blob([tar], { type: 'application/x-tar' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(tarBlob);
    a.download = bookId + '_pages.tar';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    console.log('Done! Downloaded ' + bookId + '_pages.tar (' + Math.round(tarBlob.size / 1024 / 1024) + ' MB)');
})();
