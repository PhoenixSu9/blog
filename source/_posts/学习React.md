---
title: 学习vue
date: 2025-11-12 08:04:18
tags: [React]
categories: [前端, React]
description: 系统的学习一下前端的内容，目标: 半个月完成
---

前10节课程主要包括 React的概况和常用的语法规则。涉及的还是规则比较多，以及一些React基础依赖的用法。

第一个关键点: JSX

JSX的语法规则:
1. 定义虚拟DOM时，不要写引号。
2. 标签中如果混入JS代码的时候，**表达式**要用{}括起来，*这一块不能混入非表达式的JS代码。*
3. 样式的类名的指定不要用class,要用className。
4. 内联样式，要用 style={{key:value}}的形式来写。
5. 外层只有一个根标签。内部标签必须闭合。
6. 标签的首字母有如下规则:
   1. 若小写字母开头，则该标签转为html中同名元素，若html中无该标签对应的同名元素，则报错。
   2. 所大写字母开头，react就去渲染对应的组件，若组件没有定义，则报错。

```html
<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<title>jsx语法规则</title>
	<style>
		.title{
			background-color: orange;
			width: 200px;
		}
	</style>
</head>
<body>
	<!-- 准备好一个“容器” -->
	<div id="test"></div>

	<!-- 引入react核心库 -->
	<script type="text/javascript" src="../js/react.development.js"></script>
	<!-- 引入react-dom，用于支持react操作DOM -->
	<script type="text/javascript" src="../js/react-dom.development.js"></script>
	<!-- 引入babel，用于将jsx转为js -->
	<script type="text/javascript" src="../js/babel.min.js"></script>

    <!-- 关键的内容 -->
    <script type="text/babel">
        const myId = "abc"
        const myData = "hello"
        const VDOM = (
            <div>
                <h2 className="title" id={myId.toLowerCase()}>
                    <span style={{color: 'white', fontSize: '29px'}}>{myData.toLowerCase()}</span>
                </h2>

                <h2 className="title" id={myId.toUpperCase()}>
                    <span style={{color: 'white', fontSize: '29px'}}>{myData.toUpperCase()}</span>
                </h2>
                <input type="text"/>
            </div>
        )

        ReactDom.render(VDOM, document.getElementById('test'))


    </script>
</body>

``` 
