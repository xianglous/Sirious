// {
//     let sum = [...document.querySelectorAll(".entry-content.clear>p>strong")]
//         .filter(x => x.innerText.toLowerCase() == 'summary')[0]
//         .parentElement

//     let s = ''

//     while (sum.nextElementSibling != null
//         && !sum.nextElementSibling.childElementCount
//         && sum.nextElementSibling.children[0]?.tagName?.toLowerCase() != 'strong') {
//         sum = sum.nextElementSibling
//         s += sum.innerText + '\n'
//     }

//     console.log(s)
// }



let sList = []
document.querySelectorAll('article').forEach(article => {
    let ytb = article.querySelector('.embed-youtube')?.children[0]?.src;
    let flt = [...article.querySelectorAll(".entry-content.clear>p>strong")]
        .filter(x => x.innerText.toLowerCase() == 'summary')
        .forEach(x => {
            let sum = x.parentElement
            let s = ''
            while (sum.nextElementSibling != null
                && (sum.nextElementSibling.tagName.toLowerCase()!='p' ||!sum.nextElementSibling.childElementCount)
                && sum.nextElementSibling.children[0]?.tagName?.toLowerCase() != 'strong') {
                sum = sum.nextElementSibling
                s += sum.innerText + '\n'
            }
            sList.push([ytb, s])
        })
})

// console.log(JSON.stringify(sList,null,1))







sList.forEach(x => {
    x[0] = 'https://savesubs.com/process?url=' + x[0]
})





let a = document.createElement('a')
a.href = window.URL.createObjectURL(
    new Blob([JSON.stringify(sList,null,2)], { type: 'text/json' }))
a.setAttribute('download', '')
a.click();









// https://savesubs.com/process?url=
[...[...document.querySelector("main main .list-reset.leading-normal").children]
    .filter(x => x.children[0].innerText.toLowerCase() == 'english')[0]
    .children[1].children[0].children[0].children]
    .filter(x => x.innerText.toLowerCase() == 'txt')[0]
    .click()

